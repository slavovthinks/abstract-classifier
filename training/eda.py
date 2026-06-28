"""Exploratory data analysis: stream the arXiv snapshot and report the class
distribution and abstract-length percentiles.

Streams the ~4 GB JSON-lines file one record at a time (never loads it into
memory). Writes a CSV + markdown table and two plots under ``artifacts/eda/``
for the README, and prints a summary used to choose ``PER_CLASS_CAP``.

Run with: ``make eda`` or ``uv run python -m training.eda``.
"""

import json
import logging
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arxiv_ml.enums import Category
from training import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("training.eda")

_PROGRESS_EVERY = 250_000
_PERCENTILES = (50, 90, 95, 99, 100)


def _iter_records(path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def run() -> None:
    if not config.DATA_FILE.exists():
        raise SystemExit(
            f"Dataset not found at {config.DATA_FILE}. Run `make dataset` first "
            "(requires Kaggle credentials at ~/.kaggle/kaggle.json)."
        )

    counts: Counter[Category] = Counter()
    char_lengths: list[int] = []
    word_lengths: list[int] = []
    total = 0
    dropped = 0

    for record in _iter_records(config.DATA_FILE):
        total += 1
        group = config.map_to_group(record.get("categories", ""))
        if group is None:
            dropped += 1
            continue
        counts[group] += 1
        abstract = record.get("abstract", "") or ""
        char_lengths.append(len(abstract))
        word_lengths.append(len(abstract.split()))
        if total % _PROGRESS_EVERY == 0:
            logger.info("processed %d records (%d dropped)", total, dropped)

    logger.info("done: %d records, %d mapped, %d dropped", total, total - dropped, dropped)

    config.EDA_DIR.mkdir(parents=True, exist_ok=True)
    _write_distribution(counts, total, dropped)
    _write_length_stats(np.array(char_lengths), np.array(word_lengths))
    _plot_distribution(counts)
    _plot_length_hist(np.array(char_lengths))
    logger.info("EDA artifacts written to %s", config.EDA_DIR)


def _write_distribution(counts: Counter[Category], total: int, dropped: int) -> None:
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    mapped = total - dropped
    lines = ["| category | count | share |", "|---|---:|---:|"]
    csv_lines = ["category,count,share"]
    for category, count in ordered:
        share = count / mapped if mapped else 0.0
        lines.append(f"| {category.value} | {count:,} | {share:.2%} |")
        csv_lines.append(f"{category.value},{count},{share:.6f}")
    table = "\n".join(lines)
    (config.EDA_DIR / "class_distribution.md").write_text(table + "\n")
    (config.EDA_DIR / "class_distribution.csv").write_text("\n".join(csv_lines) + "\n")
    logger.info(
        "class distribution (%d mapped, %d dropped):\n%s", mapped, dropped, table
    )


def _write_length_stats(char_lengths: np.ndarray, word_lengths: np.ndarray) -> None:
    lines = ["| metric | " + " | ".join(f"p{p}" for p in _PERCENTILES) + " |"]
    lines.append("|---|" + "---:|" * len(_PERCENTILES))
    for name, arr in (("chars", char_lengths), ("words", word_lengths)):
        pct = np.percentile(arr, _PERCENTILES)
        lines.append("| " + name + " | " + " | ".join(f"{v:,.0f}" for v in pct) + " |")
    table = "\n".join(lines)
    (config.EDA_DIR / "abstract_length_percentiles.md").write_text(table + "\n")
    logger.info("abstract length percentiles:\n%s", table)


def _plot_distribution(counts: Counter[Category]) -> None:
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [c.value for c, _ in ordered]
    values = [n for _, n in ordered]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_title("arXiv abstracts per top-level category group")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(config.EDA_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)


def _plot_length_hist(char_lengths: np.ndarray) -> None:
    upper = np.percentile(char_lengths, 99)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(char_lengths[char_lengths <= upper], bins=60)
    ax.set_title("Abstract length (chars, <= p99)")
    ax.set_xlabel("characters")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(config.EDA_DIR / "abstract_length_hist.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()