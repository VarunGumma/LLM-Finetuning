import os
from sys import argv
from pathlib import Path
from datasets import load_dataset


def prepare_dataset_properly(task):
    dataset_list = [f"raw_data/{task}_data.jsonl"]
    ds = load_dataset("json", data_files=dataset_list)["train"]
    ds = ds.train_test_split(test_size=0.01, shuffle=True, seed=42) 

    train_path, test_path = f"processed_data/{task}/train", f"processed_data/{task}/test"
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(test_path).mkdir(parents=True, exist_ok=True)

    ds["train"].save_to_disk(train_path)
    ds["test"].save_to_disk(test_path)


if __name__ == "__main__":
    prepare_dataset_properly(argv[1])