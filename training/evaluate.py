"""Shared evaluation metrics for the training scripts.

Reports accuracy and **macro** precision/recall/F1 (macro because the classes
are imbalanced — micro/accuracy flatter the majority groups and hide weak
minority classes), plus a confusion matrix. Emits a markdown table and a PNG so
the numbers can drop straight into the README.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

logger = logging.getLogger("training.evaluate")


def evaluate(y_true, y_pred, labels, out_dir: Path, split_name: str) -> dict:
    """Compute metrics for one split, write a report + confusion matrix, return a summary."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, digits=3
    )
    summary = (
        f"{split_name}: accuracy={accuracy:.3f} "
        f"macro_precision={precision:.3f} macro_recall={recall:.3f} macro_f1={f1:.3f}"
    )
    logger.info("%s\n%s", summary, report)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_report(out_dir / f"metrics_{split_name}.md", summary, report)
    _plot_confusion(y_true, y_pred, labels, out_dir / f"confusion_{split_name}.png", split_name)

    return {
        "split": split_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def _write_report(path: Path, summary: str, report: str) -> None:
    path.write_text(f"# {summary}\n\n```\n{report}\n```\n")


def _plot_confusion(y_true, y_pred, labels, path: Path, split_name: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, xticks_rotation=45, values_format=".2f", colorbar=False)
    ax.set_title(f"Confusion matrix ({split_name}, row-normalized)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)