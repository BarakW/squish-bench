from pathlib import Path
from typing import cast

from datasets import load_dataset, DatasetDict

MIN_CHARS = 8_000
HF_DATASET = "allenai/c4"
HF_CONFIG = "realnewslike"
DATASET_PATH = Path("datasets") / "c4_realnewslike_8k"


def keep_long_doc(example: dict[str, str]) -> bool:
    return len(example.get("text") or "") >= MIN_CHARS


if __name__ == "__main__":
    dataset = cast(DatasetDict, load_dataset(HF_DATASET, HF_CONFIG))
    filtered = dataset.filter(keep_long_doc, num_proc=4)

    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    filtered.save_to_disk(DATASET_PATH)
    print("Done.")
