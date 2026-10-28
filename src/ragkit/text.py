from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the "
    "this to was what when where which who why with do does can into than then".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
