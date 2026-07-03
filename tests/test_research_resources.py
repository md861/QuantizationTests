"""Tests for tracked research/evaluation resources."""

from __future__ import annotations

from pathlib import Path

from experiments.transformer_experiment import load_text_batch


def test_wikitext2_validation_256_resource_loads_expected_count():
    path = Path("docs/research_resources/eval_texts/wikitext2_raw_validation_256.txt")

    texts = load_text_batch(path)
    header = path.read_text(encoding="utf-8").splitlines()[:16]

    assert len(texts) == 256
    assert "# Evaluation record count: 256" in header
    assert "# Evaluation text word count: 23742" in header
    assert "# Evaluation text UTF-8 byte count: 125852" in header
