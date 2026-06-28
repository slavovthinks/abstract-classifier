"""Build the training dataset from the raw arXiv snapshot.

Streams the JSON-lines file, maps each record's primary category to a top-level
:class:`~arxiv_ml.enums.Category`, takes a **capped stratified subsample** via
per-class reservoir sampling (majority classes truncated to ``PER_CLASS_CAP``,
minority classes kept whole), then writes stratified train/val/test parquet
splits. Reservoir sampling keeps the pass single and memory-bounded.

Run with: ``make prepare-data`` or ``uv run python -m training.prepare_data``.
"""

import json
import logging
import random
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

from arxiv_ml.enums import Category
from training import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("training.prepare_data")

_PROGRESS_EVERY = 250_000


def _iter_records(path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _subsample() -> pd.DataFrame:
    """Per-class reservoir sample of (abstract, label), capped at PER_CLASS_CAP."""
    rng = random.Random(config.SEED)
    cap = config.PER_CLASS_CAP
    reservoir: dict[Category, list[str]] = {c: [] for c in Category}
    seen: Counter[Category] = Counter()
    total = 0

    for record in _iter_records(config.DATA_FILE):
        total += 1
        if total % _PROGRESS_EVERY == 0:
            logger.info("scanned %d records", total)

        group = config.map_to_group(record.get("categories", ""))
        if group is None:
            continue
        abstract = (record.get("abstract") or "").strip()
        if not abstract:
            continue

        seen[group] += 1
        bucket = reservoir[group]
        if len(bucket) < cap:
            bucket.append(abstract)
        else:
            j = rng.randint(1, seen[group])
            if j <= cap:
                bucket[j - 1] = abstract

    rows = [
        {"abstract": abstract, "label": category.value}
        for category, bucket in reservoir.items()
        for abstract in bucket
    ]
    rng.shuffle(rows)
    df = pd.DataFrame(rows, columns=["abstract", "label"])
    logger.info(
        "subsampled %d rows from %d records; per-class kept: %s",
        len(df),
        total,
        {c.value: len(b) for c, b in reservoir.items()},
    )
    return df


def _split_and_write(df: pd.DataFrame) -> None:
    train, temp = train_test_split(
        df,
        test_size=config.VAL_FRACTION + config.TEST_FRACTION,
        stratify=df["label"],
        random_state=config.SEED,
    )
    rel_test = config.TEST_FRACTION / (config.VAL_FRACTION + config.TEST_FRACTION)
    val, test = train_test_split(
        temp,
        test_size=rel_test,
        stratify=temp["label"],
        random_state=config.SEED,
    )

    config.PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        path = config.PREPARED_DIR / f"{name}.parquet"
        split.reset_index(drop=True).to_parquet(path, index=False)
        logger.info("wrote %s rows to %s", len(split), path)


def run() -> None:
    if not config.DATA_FILE.exists():
        raise SystemExit(
            f"Dataset not found at {config.DATA_FILE}. Run `make dataset` first "
            "(requires Kaggle credentials at ~/.kaggle/kaggle.json)."
        )
    df = _subsample()
    _split_and_write(df)
    logger.info("prepared splits written to %s", config.PREPARED_DIR)


if __name__ == "__main__":
    run()