"""Classical text preprocessing for the TF-IDF baseline.

This lives in ``arxiv_ml`` (not ``training``) on purpose: the fitted TF-IDF
pipeline pickles a *reference* to :func:`clean`, so training and serving must
import the exact same function. Keeping it here means the serving layer never
depends on ``training`` and there is no train/inference preprocessing drift.

Stopwords are removed and tokens stemmed *here*, so the ``TfidfVectorizer`` is
configured with this as its ``preprocessor`` and without its own ``stop_words``
(which would otherwise run against already-stemmed tokens).
"""

import re

import snowballstemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z]+")
_MIN_TOKEN_LEN = 3
_stemmer = snowballstemmer.stemmer("english")


def clean(text: str) -> str:
    """Lowercase, strip non-alpha, drop English stopwords, and Porter-stem.

    Returns a space-joined string of stemmed tokens, ready for the vectorizer's
    default tokenizer.
    """
    tokens = [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= _MIN_TOKEN_LEN and token not in ENGLISH_STOP_WORDS
    ]
    return " ".join(_stemmer.stemWords(tokens))
