"""Finance-headline lexicon scorer: the sentiment agent's offline fallback.

Used when no LLM is configured or the LLM call fails. It improves on the draft's bag of
substrings (where "high" matched "highway" and "not a beat" counted as bullish) with:

* **tokenisation + light lemmatisation** — "beats", "beat", "beating" all hit ``beat``;
* **weighted terms** — "fraud" (−2.5) is worse news than "concern" (−0.8);
* **phrases** — multi-word expressions ("raises guidance", "misses estimates",
  "record high") score as a unit and override their individual words. Up to
  :data:`PHRASE_GAP` filler words may sit between phrase words ("raises *full-year*
  guidance", "beats *earnings* estimates");
* **negation** — a negator ("not", "no", "never", "without", "n't" contractions, …)
  within :data:`NEGATION_WINDOW` tokens before a term flips and damps it
  (×:data:`NEGATION_FACTOR`): "did not beat estimates" is bearish. Negation never reaches
  across punctuation or a conjunction ("misses estimates *but* raises guidance");
* **"miss" as a negator** — "miss" (and "fail") is bearish on its own *and* negates the
  positive term that follows it: "misses profit forecast" is worse than "misses", and
  "fails to win approval" is bearish.

A headline's raw score ``Σ term weights`` is squashed to ``tanh(raw / 2)`` ∈ (−1, 1).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final

#: Maximum number of filler tokens allowed between consecutive words of a phrase.
PHRASE_GAP: Final = 2
#: How many tokens back a negator reaches.
NEGATION_WINDOW: Final = 3
#: Negated terms flip sign and lose some strength ("not bad" ≠ "good").
NEGATION_FACTOR: Final = -0.8
#: ``score = tanh(raw / SQUASH_SCALE)``: a single strong term (≈ ±2) scores ≈ ±0.76.
SQUASH_SCALE: Final = 2.0

# Word lemmas → weight. Inflections are resolved by :func:`lemmatize`.
WORD_WEIGHTS: Final[Mapping[str, float]] = {
    # ---------------------------------------------------------------- bullish
    "beat": 1.5,
    "surge": 1.5,
    "soar": 1.6,
    "rally": 1.3,
    "jump": 1.1,
    "climb": 0.9,
    "gain": 0.8,
    "rise": 0.7,
    "rose": 0.7,
    "rebound": 1.0,
    "recover": 0.8,
    "recovery": 0.9,
    "boost": 0.9,
    "accelerate": 0.9,
    "expand": 0.6,
    "growth": 0.8,
    "grow": 0.7,
    "strong": 0.8,
    "stronger": 0.9,
    "robust": 0.8,
    "record": 0.6,
    "profit": 0.6,
    "profitable": 0.9,
    "upgrade": 1.5,
    "outperform": 1.2,
    "bullish": 1.5,
    "buy": 0.5,
    "buyback": 1.2,
    "buy-back": 1.2,
    "dividend": 0.5,
    "win": 0.9,
    "approval": 1.0,
    "approve": 0.9,
    "breakthrough": 1.2,
    "exceed": 1.2,
    "upbeat": 1.0,
    "optimistic": 0.9,
    "higher": 0.4,
    "high": 0.3,
    "positive": 0.7,
    "raise": 0.5,
    "partnership": 0.6,
    "demand": 0.3,
    # ---------------------------------------------------------------- bearish
    "miss": -1.5,
    "plunge": -1.8,
    "crash": -2.0,
    "tumble": -1.5,
    "sink": -1.3,
    "slump": -1.4,
    "slide": -1.0,
    "drop": -1.0,
    "fall": -0.9,
    "fell": -0.9,
    "decline": -1.0,
    "slow": -0.6,
    "slowdown": -1.0,
    "weak": -0.9,
    "weaker": -0.9,
    "weaken": -0.9,
    "loss": -1.0,
    "lose": -0.8,
    "downgrade": -1.5,
    "underperform": -1.2,
    "bearish": -1.5,
    "sell": -0.5,
    "selloff": -1.5,
    "sell-off": -1.5,
    "cut": -0.8,
    "slash": -1.5,
    "layoff": -1.3,
    "lawsuit": -1.5,
    "sue": -1.2,
    "probe": -1.2,
    "investigation": -1.2,
    "fraud": -2.5,
    "scandal": -2.0,
    "bankruptcy": -3.0,
    "bankrupt": -3.0,
    "default": -1.8,
    "recession": -1.2,
    "warn": -1.2,
    "warning": -1.2,
    "concern": -0.8,
    "risk": -0.4,
    "debt": -0.5,
    "recall": -1.0,
    "delay": -0.7,
    "halt": -1.0,
    "fined": -1.0,
    "penalty": -1.0,
    "downturn": -1.2,
    "lower": -0.4,
    "low": -0.3,
    "negative": -0.7,
    "volatile": -0.3,
    "shortfall": -1.3,
    "disappoint": -1.3,
    "pessimistic": -0.9,
    "headwind": -0.8,
}

# Phrases (tuples of lemmas) → weight; they replace the weights of their words.
PHRASE_WEIGHTS: Final[Mapping[tuple[str, ...], float]] = {
    ("beat", "estimate"): 2.0,
    ("beat", "expectation"): 2.0,
    ("top", "estimate"): 1.8,
    ("top", "expectation"): 1.8,
    ("exceed", "expectation"): 2.0,
    ("raise", "guidance"): 2.0,
    ("raise", "outlook"): 1.8,
    ("raise", "forecast"): 1.8,
    ("raise", "dividend"): 1.5,
    ("dividend", "increase"): 1.3,
    ("share", "buyback"): 1.5,
    ("price", "target", "raise"): 1.3,
    ("record", "high"): 1.6,
    ("record", "quarter"): 1.4,
    ("record", "revenue"): 1.4,
    ("record", "profit"): 1.6,
    ("all-time", "high"): 1.6,
    ("upgrade", "buy"): 1.8,
    ("strong", "demand"): 1.3,
    ("miss", "estimate"): -2.0,
    ("miss", "expectation"): -2.0,
    ("fall", "short"): -1.6,
    ("cut", "guidance"): -2.0,
    ("slash", "guidance"): -2.5,
    ("lower", "guidance"): -2.0,
    ("cut", "outlook"): -1.8,
    ("cut", "forecast"): -1.8,
    ("cut", "dividend"): -1.8,
    ("profit", "warning"): -2.2,
    ("record", "low"): -1.6,
    ("record", "loss"): -1.8,
    ("downgrade", "sell"): -1.8,
    ("weak", "demand"): -1.3,
    ("cut", "cost"): 1.0,
    ("cost", "fall"): 0.8,
    ("cost", "rise"): -0.8,
    ("job", "cut"): -1.3,
    ("going", "concern"): -2.5,
    ("price", "target", "cut"): -1.3,
}

NEGATORS: Final = frozenset(
    {"not", "no", "never", "without", "nor", "neither", "none", "hardly", "barely", "lack"}
)
#: Words that end a negation's scope ("X misses estimates but raises guidance").
SCOPE_BREAKERS: Final = frozenset(
    {"but", "while", "although", "though", "yet", "and", "as", "after", "despite", "however"}
)
#: Bearish words that also negate the *positive* term that follows them.
NEGATING_TERMS: Final = frozenset({"miss", "fail"})

_IRREGULAR: Final[Mapping[str, str]] = {
    "fell": "fell",
    "rose": "rose",
    "won": "win",
    "wins": "win",
    "winning": "win",
    "sold": "sell",
    "selling": "sell",
    "lost": "lose",
    "cuts": "cut",
    "cutting": "cut",
    "beats": "beat",
    "beating": "beat",
    "tops": "top",
    "topped": "top",
    "topping": "top",
    "estimates": "estimate",
    "expectations": "expectation",
    "forecasts": "forecast",
    "shares": "share",
    "jobs": "job",
    "targets": "target",
    "quarters": "quarter",
    "losses": "loss",
    "concerns": "concern",
    "lawsuits": "lawsuit",
    "layoffs": "layoff",
    "sued": "sue",
    "sues": "sue",
}
_SUFFIXES: Final = ("ing", "ed", "es", "s", "d")
_TOKEN_RE: Final = re.compile(r"[a-z0-9][a-z0-9'\-]*")
#: Punctuation that separates clauses (phrases and negation never cross it).
_CLAUSE_RE: Final = re.compile(r"[,;:!?()\[\]|]|\.(?:\s|$)|\s[-–—]\s")
#: Lemmas used only inside phrases (no weight on their own).
_PHRASE_ONLY: Final = frozenset(word for phrase in PHRASE_WEIGHTS for word in phrase) - set(
    WORD_WEIGHTS
)
_VOCABULARY: Final = frozenset(WORD_WEIGHTS) | _PHRASE_ONLY | NEGATING_TERMS


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; ``n't`` contractions become a separate ``not`` token."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower().replace("’", "'")):
        token = raw.strip("'-")
        if not token:
            continue
        if token.endswith("n't"):
            stem = token[:-3]
            if stem:
                tokens.append(stem)
            tokens.append("not")
            continue
        if token.endswith("'s"):
            token = token[:-2]
        tokens.append(token)
    return tokens


def lemmatize(token: str) -> str:
    """Map an inflected token onto a lexicon lemma when one matches (else unchanged).

    Handles regular inflections (``-s/-es/-ed/-d/-ing``, ``-ies/-ied`` → ``-y``,
    dropped final ``e``, doubled final consonant) plus a few irregular forms.
    """
    if token in _VOCABULARY:
        return token
    if token in _IRREGULAR:
        return _IRREGULAR[token]
    if token.endswith(("ies", "ied")) and len(token) > 4:
        candidate = token[:-3] + "y"
        if candidate in _VOCABULARY:
            return candidate
    for suffix in _SUFFIXES:
        if not token.endswith(suffix) or len(token) <= len(suffix) + 2:
            continue
        stem = token[: -len(suffix)]
        for candidate in (stem, stem + "e", stem[:-1] if stem[-1:] == stem[-2:-1] else ""):
            if candidate and candidate in _VOCABULARY:
                return candidate
    return token


@dataclass(frozen=True)
class TermHit:
    """One scored term (word or phrase) inside a headline."""

    term: str
    weight: float
    negated: bool = False

    def describe(self) -> str:
        prefix = "NOT " if self.negated else ""
        return f"{prefix}{self.term} {self.weight:+.1f}"


@dataclass(frozen=True)
class HeadlineScore:
    """Lexicon reading of one headline."""

    text: str
    score: float
    raw: float
    hits: tuple[TermHit, ...] = field(default_factory=tuple)

    @property
    def has_signal(self) -> bool:
        return bool(self.hits) and self.raw != 0.0

    @property
    def label(self) -> str:
        if self.score > 0.1:
            return "bullish"
        if self.score < -0.1:
            return "bearish"
        return "neutral"


def _match_phrase(lemmas: list[str], start: int, phrase: tuple[str, ...]) -> list[int] | None:
    """Token positions of ``phrase`` starting at ``start`` (gaps ≤ PHRASE_GAP), or ``None``."""
    if lemmas[start] != phrase[0]:
        return None
    positions = [start]
    cursor = start
    for word in phrase[1:]:
        for offset in range(1, PHRASE_GAP + 2):
            j = cursor + offset
            if j < len(lemmas) and lemmas[j] == word:
                positions.append(j)
                cursor = j
                break
        else:
            return None
    return positions


def _negated(lemmas: list[str], position: int, weight: float) -> bool:
    """Is the term at ``position`` negated by a negator in the preceding window?"""
    for j in range(position - 1, max(-1, position - 1 - NEGATION_WINDOW), -1):
        lemma = lemmas[j]
        if lemma in SCOPE_BREAKERS:
            return False
        if lemma in NEGATORS:
            return True
        # "misses profit forecast": miss negates the *positive* term after it.
        if lemma in NEGATING_TERMS and weight > 0:
            return True
    return False


_PHRASES_BY_LENGTH: Final = sorted(PHRASE_WEIGHTS.items(), key=lambda kv: -len(kv[0]))


def _hit(term: str, weight: float, negated: bool) -> TermHit:
    return TermHit(term, weight * NEGATION_FACTOR if negated else weight, negated)


def _score_clause(lemmas: list[str]) -> list[TermHit]:
    """Scored terms of one clause, in reading order."""
    consumed: set[int] = set()
    found: list[tuple[int, TermHit]] = []
    # Phrases first (longest first) so they override their component words.
    for start in range(len(lemmas)):
        for phrase, weight in _PHRASES_BY_LENGTH:
            positions = _match_phrase(lemmas, start, phrase)
            if positions is None or consumed.intersection(positions):
                continue
            consumed.update(positions)
            found.append((start, _hit(" ".join(phrase), weight, _negated(lemmas, start, weight))))
            break
    for i, lemma in enumerate(lemmas):
        if i in consumed or lemma not in WORD_WEIGHTS:
            continue
        weight = WORD_WEIGHTS[lemma]
        found.append((i, _hit(lemma, weight, _negated(lemmas, i, weight))))
    return [hit for _, hit in sorted(found, key=lambda item: item[0])]


def score_headline(text: str, ignore: Iterable[str] = ()) -> HeadlineScore:
    """Score one headline in ``(−1, 1)`` (see the module docstring for the rules).

    ``ignore`` lists tokens to skip — pass the ticker so that a symbol which is also an
    English word (``LOW``, ``WIN``) does not score itself.
    """
    skip = {token.lower() for token in ignore}
    hits: list[TermHit] = []
    for clause in _CLAUSE_RE.split(text):
        lemmas = [lemmatize(token) for token in tokenize(clause) if token not in skip]
        if lemmas:
            hits.extend(_score_clause(lemmas))
    raw = sum(hit.weight for hit in hits)
    return HeadlineScore(text=text, score=math.tanh(raw / SQUASH_SCALE), raw=raw, hits=tuple(hits))


@dataclass(frozen=True)
class LexiconResult:
    """Aggregate lexicon sentiment over several headlines."""

    signal: float
    #: Fraction of headlines containing at least one sentiment term.
    coverage: float
    #: ``|Σ sᵢ| / Σ |sᵢ|`` over the headline scores (1 = unanimous direction).
    agreement: float
    headlines: tuple[HeadlineScore, ...]


def score_headlines(texts: Iterable[str], ignore: Iterable[str] = ()) -> LexiconResult:
    """Average headline score (neutral headlines count as 0) plus coverage/agreement."""
    skip = tuple(ignore)
    scores = tuple(score_headline(text, skip) for text in texts)
    if not scores:
        return LexiconResult(0.0, 0.0, 0.0, ())
    total = sum(s.score for s in scores)
    magnitude = sum(abs(s.score) for s in scores)
    coverage = sum(1 for s in scores if s.has_signal) / len(scores)
    agreement = abs(total) / magnitude if magnitude > 0 else 0.0
    return LexiconResult(
        signal=max(-1.0, min(1.0, total / len(scores))),
        coverage=coverage,
        agreement=agreement,
        headlines=scores,
    )


__all__ = [
    "NEGATION_FACTOR",
    "NEGATION_WINDOW",
    "PHRASE_GAP",
    "PHRASE_WEIGHTS",
    "SCOPE_BREAKERS",
    "WORD_WEIGHTS",
    "HeadlineScore",
    "LexiconResult",
    "TermHit",
    "lemmatize",
    "score_headline",
    "score_headlines",
    "tokenize",
]
