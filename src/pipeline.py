"""PyTorch BERT model training pipeline for the SST-2 dataset.
"""

import os
import sys
import inspect
import logging
from pprint import pprint
from typing import Annotated

import zenml
import datasets
import transformers
import evaluate
import accelerate
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from zenml.integrations.huggingface.steps import run_with_accelerate  # pylint: disable=unused-import
from zenml.materializers.materializer_registry import materializer_registry

from materializers import DatasetMaterializer, TokenizerMaterializer


CHECKPOINT = "bert-base-uncased"

materializer_registry.register_materializer_type(
    datasets.Dataset, DatasetMaterializer
)
materializer_registry.register_materializer_type(
    transformers.PreTrainedTokenizerBase, TokenizerMaterializer
)


@zenml.step
def load_data(dataset_size_percentage: int = 100) -> tuple[
    Annotated[datasets.Dataset, "training_dataset"],
    Annotated[datasets.Dataset, "validation_dataset"],
    Annotated[transformers.PreTrainedTokenizerBase, "tokenizer"]
]:
    """Load and preprocess the SST-2 dataset.

    Args:
        dataset_size_percentage: Percentage of training dataset to use (1-100)
    """

    dataset = datasets.load_dataset("glue", "sst2")

    tokenizer = transformers.AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["sentence"], truncation=True),
        batched=True,
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["idx", "sentence"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    # Calculate the number of samples to use based on percentage
    train_size = len(tokenized_dataset["train"])
    samples_to_use = int(train_size * dataset_size_percentage / 100)

    # Select the specified percentage of training data
    train_dataset = tokenized_dataset["train"].select(range(samples_to_use))

    logging.info(
        "successfully loaded and tokenized dataset with %d training samples (%.1f%% of total)",
        len(train_dataset),
        dataset_size_percentage
    )

    return train_dataset, tokenized_dataset["validation"], tokenizer


@zenml.step(enable_cache=False)
def check_dataset(
    training_dataset: datasets.Dataset,
    _validation_dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    """Print dataset information for debugging purposes."""

    logging.info("Dataset information:")
    logging.info(training_dataset)
    print("Dataset features:")
    pprint(training_dataset.features)
    print("---")
    print(tokenizer.decode(training_dataset[0]["input_ids"]))


## ZenML 0.74.0 issue:
## The accelerate decorator doesn't properly handle the custom types passed
## to accelerated steps; work around by merging the loading and training steps
## of the pipeline. Disabling accelerate by default to enable local pipeline
## runs for demonstration purposes.
#
# Defaults to the number of GPUs correctly w/o params.
# pylint: disable=too-many-locals
# @run_with_accelerate
@zenml.step(enable_cache=False)
def train_model(
    training_dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> Annotated[transformers.PreTrainedModel, "model"]:
    """Train a BERT model on the SST-2 dataset.

    Args:
        training_dataset: The preprocessed training dataset
        tokenizer: The tokenizer used for preprocessing
        
    Returns:
        The trained model
    """
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        # training_dataset.shuffle(seed=42).select(range(500)),
        training_dataset,
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=2,
    )

    batch = next(iter(train_dataloader))
    logging.info("batch shape: %s", {k: v.shape for k, v in batch.items()})
    outputs = model(**batch)
    logging.info("outputs: %s, %s", outputs.loss, outputs.logits.shape)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    ## Accelerator handles device placement automatically. The PyTorch code
    ## would have been:
    # device = torch.device("mps") if torch.mps.is_available() else torch.device("cuda")
    # model.to(device)
    # logging.info(f"device: {device}")
    accelerator = accelerate.Accelerator()
    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer)

    num_epochs = 3
    num_training_steps = len(train_dataloader) * num_epochs
    lr_scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logging.info("num_training_steps: %s", num_training_steps)

    terminal_width = int(os.popen('tput cols').read().strip())
    tqdm_width = terminal_width - len(inspect.currentframe().f_code.co_name) - 3
    progress_bar = tqdm(
        range(num_training_steps),
        file=sys.stdout,
        ncols=tqdm_width,
    )

    model.train()
    for _epoch in range(num_epochs):
        for batch in train_dataloader:
            ## placed automatically
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            # loss.backward()
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    unwrapped_model = accelerator.unwrap_model(model)
    return unwrapped_model


@zenml.step
def evaluate_model(
    model: transformers.PreTrainedModel,
    validation_dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> Annotated[dict, "metrics"]:
    """Evaluate the trained model on the validation dataset."""

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    eval_dataloader = DataLoader(
        validation_dataset,
        batch_size=8,
        collate_fn=data_collator
    )
    metric = evaluate.load("glue", "sst2")

    device = torch.device("mps") if torch.mps.is_available() else torch.device("cuda")
    model.to(device)
    logging.info("device: %s", device)

    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metrics = metric.compute()
    logging.info("metrics: %s", metrics)

    return metrics


@zenml.pipeline
def pipeline(dataset_size_percentage: int = 100) -> None:
    """Run the complete PyTorch model training pipeline.
    
    This pipeline:
    1. Loads and preprocesses the SST-2 dataset
    2. Trains a BERT model for sentiment classification
    3. Evaluates the model performance on the validation set

    Args:
        dataset_size_percentage: Percentage of training dataset to use (1-100)
    """
    # Load and preprocess data
    training_dataset, validation_dataset, tokenizer = load_data(
        dataset_size_percentage=dataset_size_percentage)
    check_dataset(training_dataset, validation_dataset, tokenizer)

    # Train the model
    model = train_model(
        training_dataset=training_dataset,
        tokenizer=tokenizer
    )

    # Evaluate the model
    evaluate_model(
        model=model,
        validation_dataset=validation_dataset,
        tokenizer=tokenizer
    )
