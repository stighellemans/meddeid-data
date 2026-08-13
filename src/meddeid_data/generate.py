"""Compatibility helpers for synthetic document generation."""

from __future__ import annotations

from .generator import generate_documents


def generate_example(index: int, *, seed: int = 42) -> dict:
    """Generate one full synthetic document using the packaged lookup lists."""

    row = generate_documents(1, seed=seed + index, require_synthea=False)[0]
    row["document_id"] = f"synthetic-{index + 1:05d}"
    return row
