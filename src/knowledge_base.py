"""SETI knowledge-base retrieval (BM25, offline, no external calls).

The kiosk's voice model (Amazon Nova Sonic) is a realtime speech-to-speech
model, so it can only receive extra context through a `@function_tool` result
(see the "External data and RAG" LiveKit guide) — there is no `on_user_turn_completed`
text hook for realtime models. The corpus is small and curated (see
../KNOWLEDGE_BASE_NOTES.md for what was included/excluded from the source PDF),
so BM25 keyword retrieval over markdown sections is enough: no embedding model,
no vector DB, no network call, fully deterministic and testable offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent.parent / "data"

_TOKEN_RE = re.compile(r"[a-záéíóúüñ0-9]+")

# High-frequency Spanish function words. Without this, generic query terms
# (and "seti" itself, which appears in nearly every chunk) drown out the
# content words that should drive BM25 ranking.
_STOPWORDS = frozenset(
    [
        "que",
        "qué",
        "es",
        "a",
        "y",
        "se",
        "el",
        "la",
        "los",
        "las",
        "de",
        "del",
        "en",
        "un",
        "una",
        "con",
        "por",
        "para",
        "su",
        "sus",
        "lo",
        "como",
        "o",
        "pero",
        "no",
        "si",
        "al",
        "este",
        "esta",
        "estos",
        "estas",
        "ese",
        "esa",
        "son",
        "ser",
        "está",
        "están",
        "cuál",
        "cuáles",
        "quién",
        "quiénes",
        "dónde",
        "cuándo",
        "tu",
        "tus",
        "le",
        "les",
        "nos",
        "sobre",
    ]
)


def _stem(word: str) -> str:
    """Strip common Spanish plural suffixes so "bancos" matches "Banco"."""
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    source: str
    heading: str
    text: str


def _split_into_chunks(path: Path) -> list[Chunk]:
    """Split a markdown file into chunks on level-2 (##) headings."""
    raw = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^## ", raw)
    chunks: list[Chunk] = []
    # sections[0] is the title (level-1 heading) plus any preamble before the
    # first "## " — keep it as its own chunk if it has real content.
    preamble = sections[0].strip()
    if preamble:
        heading = preamble.splitlines()[0].lstrip("# ").strip()
        chunks.append(Chunk(source=path.name, heading=heading, text=preamble))
    for section in sections[1:]:
        section = section.strip()
        if not section:
            continue
        heading = section.splitlines()[0].strip()
        chunks.append(Chunk(source=path.name, heading=heading, text=f"## {section}"))
    return chunks


@lru_cache(maxsize=1)
def _load_chunks() -> tuple[Chunk, ...]:
    files = sorted(DATA_DIR.glob("*.md"))
    chunks: list[Chunk] = []
    for file in files:
        chunks.extend(_split_into_chunks(file))
    return tuple(chunks)


@lru_cache(maxsize=1)
def _bm25_index() -> tuple[BM25Okapi, tuple[Chunk, ...]]:
    chunks = _load_chunks()
    corpus = [_tokenize(chunk.text) for chunk in chunks]
    return BM25Okapi(corpus), chunks


# The two chunks that best answer "what is SETI / what does it do". Always
# included: a broad conversational question ("cuéntame de SETI", "qué sabes
# de la empresa") carries almost no lexical signal — "seti" itself appears in
# nearly every chunk, so BM25 alone can bury this content behind weaker or
# off-target matches (see huella-guide logs, 2026-09-02 session, where the
# guide ended up narrating the tool's perceived gaps instead of speaking
# confidently). Pinning these keeps every answer grounded in solid material.
_ANCHOR_KEYS = (
    ("01-identidad-posicionamiento.md", "Identidad"),
    ("01-identidad-posicionamiento.md", "Propuesta de valor central"),
)
_MAX_CHUNKS = 5


def search_seti_knowledge(query: str, top_k: int = 4) -> str:
    """Return the most relevant knowledge-base chunks for a query, formatted
    as plain text for a voice model to paraphrase (not read verbatim)."""
    bm25, chunks = _bm25_index()
    if not chunks:
        return "La base de conocimiento de SETI está vacía."

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    matched = [i for i in ranked if scores[i] > 0][:top_k]
    anchors = [i for i, c in enumerate(chunks) if (c.source, c.heading) in _ANCHOR_KEYS]

    selected = list(dict.fromkeys(anchors + matched))[:_MAX_CHUNKS]

    if not selected:
        return (
            "No hay información específica sobre esa consulta en la base de "
            "conocimiento de SETI."
        )

    return "\n\n---\n\n".join(chunks[i].text for i in selected)
