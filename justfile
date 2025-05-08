lint:
    poetry run pylint src/ --output-format colorized

run:
    poetry run python src/run.py --dataset-size 1
