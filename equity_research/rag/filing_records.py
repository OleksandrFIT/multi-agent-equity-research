from __future__ import annotations

import hashlib
from datetime import date

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

# ~500 tokens ≈ ~2000 chars per child chunk for precise matching.
_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)


class FilingSection(BaseModel):
    ticker: str
    section: str  # "risk_factors" | "mda"
    filed_at: date
    form: str
    text: str


def filing_parent_id(section: FilingSection) -> str:
    raw = f"{section.ticker}|{section.form}|{section.section}|{section.filed_at.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def filing_metadata(section: FilingSection) -> dict:
    return {
        "doc_type": "filing_section",
        "ticker": section.ticker,
        "date_int": int(section.filed_at.strftime("%Y%m%d")),
        "form": section.form,
        "section": section.section,
        "content_hash": filing_parent_id(section),
    }


def chunk_filing(section: FilingSection) -> tuple[str, str, list[tuple[str, dict]]]:
    pid = filing_parent_id(section)
    base = filing_metadata(section)
    children = _splitter.split_text(section.text) or [section.text]
    child_chunks = [(c, {**base, "parent_id": pid}) for c in children]
    return pid, section.text, child_chunks
