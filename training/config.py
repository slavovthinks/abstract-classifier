"""Shared, parameterized knobs for the training pipeline.

Imports only ``arxiv_ml.enums`` (the dependency direction is training -> arxiv_ml,
never the reverse) and the stdlib, so it stays framework-agnostic.
"""

from pathlib import Path

from arxiv_ml.enums import Category

# --- Reproducibility & subsampling -----------------------------------------

SEED = 42

# Per-class cap for the capped stratified subsample: majority classes are
# truncated to this many rows, minority classes are kept whole. Chosen from EDA
# (~3.08M records, 126:1 raw imbalance): 20k yields ~145k total rows at ~1.7:1
# imbalance, with q-fin (13k) and econ (12k) kept whole.
PER_CLASS_CAP = 20_000

VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "arxiv-metadata-oai-snapshot.json"
PREPARED_DIR = DATA_DIR / "prepared"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "tfidf"
EDA_DIR = PROJECT_ROOT / "artifacts" / "eda"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"

# --- arXiv taxonomy -> top-level Category group ----------------------------

# arXiv's "archive" is the part of a category before the dot (``cs.LG`` -> ``cs``,
# ``astro-ph.GA`` -> ``astro-ph``, ``hep-ph`` -> ``hep-ph``). We collapse the
# physics-family archives into a single PHYSICS bucket; the other archives map
# 1:1 onto a Category. Archives not listed here (rare legacy tags) are dropped.

_DIRECT: dict[str, Category] = {
    "cs": Category.CS,
    "math": Category.MATH,
    "q-bio": Category.Q_BIO,
    "q-fin": Category.Q_FIN,
    "stat": Category.STAT,
    "eess": Category.EESS,
    "econ": Category.ECON,
}

_PHYSICS_ARCHIVES: frozenset[str] = frozenset(
    {
        # current physics-group archives
        "astro-ph",
        "cond-mat",
        "gr-qc",
        "hep-ex",
        "hep-lat",
        "hep-ph",
        "hep-th",
        "math-ph",
        "nlin",
        "nucl-ex",
        "nucl-th",
        "physics",
        "quant-ph",
        # legacy / discontinued physics archives still present in old records
        "acc-phys",
        "adap-org",
        "ao-sci",
        "atom-ph",
        "bayes-an",
        "chao-dyn",
        "comp-gas",
        "mtrl-th",
        "patt-sol",
        "plasm-ph",
        "solv-int",
        "supr-con",
    }
)


def archive_of(primary_category: str) -> str:
    """Return the archive (text before the first dot) of a single category tag."""
    return primary_category.split(".", 1)[0]


def map_to_group(categories: str) -> Category | None:
    """Map an arXiv ``categories`` field to its top-level Category group.

    Uses the *primary* (first-listed) category. Returns ``None`` for records
    whose primary archive is outside our label scheme (these are dropped).
    """
    primary = categories.split(" ", 1)[0]
    archive = archive_of(primary)
    if archive in _PHYSICS_ARCHIVES:
        return Category.PHYSICS
    return _DIRECT.get(archive)