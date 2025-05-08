"""Main entry point for running the BERT training pipeline.

Example:
    Run with default settings (100% of dataset):
        python src/run.py

    Run with 50% of the dataset:
        python src/run.py --dataset-size 50

The script can be run in two modes:
1. Local execution: Uses local GPU (MPS) if available
2. AWS execution: Uses configured AWS instance with GPU support (requires
   remote ZenML server with a Skypilot stack)
Switch between the modes by running:
    zenml stack set <stack-name>

Note:
    The appropriate settings for the execution environment are applied
    automatically. Docker and AWS settings are ignored on the local (default)
    stack.
"""

import click

import settings
from pipeline import pipeline


@click.command()
@click.option('--dataset-size', type=click.IntRange(1, 100), default=100,
              help='Percentage of training dataset to use (1-100)')
def main(dataset_size: int) -> None:
    """Train BERT model on SST-2 dataset.

    Args:
        dataset_size: Percentage of training dataset to use (1-100)
    """
    # Configure pipeline settings
    pipeline_settings = {
        # Docker settings (ignored if Docker is not built)
        "docker": settings.docker,
        # AWS VM settings (ignored if AWS VM is not part of the stack)
        "orchestrator.vm_aws": settings.skypilot
    }

    # Run the pipeline with configured settings
    pipeline.with_options(settings=pipeline_settings)(
        dataset_size_percentage=dataset_size)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
