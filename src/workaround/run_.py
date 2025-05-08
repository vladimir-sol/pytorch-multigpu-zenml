"""PyTorch BERT model training pipeline for MRPC dataset."""
# pylint: disable=duplicate-code

import os
import sys
import inspect
import logging
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
from zenml.integrations.huggingface.steps import run_with_accelerate
from zenml.materializers.materializer_registry import materializer_registry

import settings
from materializers import DatasetMaterializer, TokenizerMaterializer


CHECKPOINT = "bert-base-uncased"

materializer_registry.register_materializer_type(
    datasets.Dataset, DatasetMaterializer
)
materializer_registry.register_materializer_type(
    transformers.PreTrainedTokenizerBase, TokenizerMaterializer
)


# ZenML 0.74.0 has an issue with passing custom types to accelerated steps;
# workaround by merging the loading and training steps.
# pylint: disable=too-many-locals
@run_with_accelerate  # defaults to the number of GPUs correctly w/o params
@zenml.step(enable_cache=False)
def train_model(
    # training_dataset: datasets.Dataset,
    # tokenizer: transformers.PreTrainedTokenizerBase,
) -> tuple[
    Annotated[transformers.PreTrainedModel, "model"],
    Annotated[dict, "metrics"]
]:
    """Load, train and evaluate in a single step."""

    #
    # load_data()
    #

    dataset = datasets.load_dataset("glue", "sst2")

    tokenizer = transformers.AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["sentence"], truncation=True),
        # Note: padding="max_length" slows down training from 1m20s to 6m+
        batched=True,
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["idx", "sentence"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    logging.info(
        "successfully loaded and tokenized dataset with %s training samples",
        len(tokenized_dataset['train'])
    )
    training_dataset = tokenized_dataset["train"]
    validation_dataset = tokenized_dataset["validation"]

    #
    # train_model()
    #

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
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

    ## Accelerator handles device placement automatically
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

    #
    # evaluate_model()
    #

    eval_dataloader = DataLoader(
        validation_dataset,
        batch_size=8,
        collate_fn=data_collator
    )
    metric = evaluate.load("glue", "sst2")
    eval_dataloader, metric = accelerator.prepare(
        eval_dataloader, metric)

    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metrics = metric.compute()
    logging.info("metrics: %s", metrics)

    unwrapped_model = accelerator.unwrap_model(model)
    return unwrapped_model, metrics


@zenml.pipeline
def pipeline() -> None:
    """Run the complete PyTorch model training pipeline.
    
    This pipeline:
    1. Loads and preprocesses the data
    2. Trains a BERT model
    3. Evaluates the model performance
    """
    ## Load data
    # training_dataset, validation_dataset, tokenizer = load_data()
    # check_dataset(training_dataset, validation_dataset, tokenizer)

    ## Train model
    # model = train_model(training_dataset=training_dataset, tokenizer=tokenizer)
    train_model()

    ## Evaluate model
    # evaluate_model(model, validation_dataset, tokenizer)


if __name__ == "__main__":
    pipeline.with_options(
        settings={
            # Ignored if Docker is not built (i.e. local):
            # https://docs.zenml.io/concepts/containerization
            "docker": settings.docker,

            # Ignored if AWS VM is not a part of the stack (i.e. local):
            # https://docs.zenml.io/concepts/steps_and_pipelines/configuration#using-the-right-key-for-stack-component-settings
            "orchestrator.vm_aws": settings.skypilot
        }
    )()
