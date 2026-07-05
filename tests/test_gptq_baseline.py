"""Tests for the optional GPTQ external baseline runner."""

from __future__ import annotations

import argparse
import csv

import numpy as np
import pytest
import torch

from experiments import gptq_baseline as gptq


def _args(**overrides):
    defaults = dict(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        gptq_model_name="test/tinyllama-gptq",
        eval_text_file=None,
        max_eval_texts=None,
        results_dir=None,
        local_files_only=False,
        device="auto",
        torch_dtype="float16",
        trust_remote_code=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_config_uses_eval_text_file(tmp_path):
    text_path = tmp_path / "eval.txt"
    text_path.write_text("First text.\n\nSecond text.\n", encoding="utf-8")

    config = gptq.build_config(
        _args(
            eval_text_file=text_path,
            max_eval_texts=1,
            results_dir=tmp_path / "results",
            local_files_only=True,
            torch_dtype="bfloat16",
            trust_remote_code=True,
        )
    )

    assert config.evaluation_texts == ["First text."]
    assert config.evaluation_text_source == str(text_path)
    assert config.results_dir == tmp_path / "results"
    assert config.local_files_only is True
    assert config.torch_dtype == "bfloat16"
    assert config.trust_remote_code is True
    assert config.gptq_model_name == "test/tinyllama-gptq"


def test_resolve_torch_dtype_accepts_supported_values():
    assert gptq._resolve_torch_dtype("float16") is torch.float16
    assert gptq._resolve_torch_dtype("bfloat16") is torch.bfloat16
    assert gptq._resolve_torch_dtype("float32") is torch.float32


def test_resolve_torch_dtype_rejects_unknown_value():
    with pytest.raises(ValueError, match="torch dtype"):
        gptq._resolve_torch_dtype("float8")


def test_make_logit_record_computes_shared_metrics():
    config = gptq.GptqBaselineConfig(
        model_name="test-model",
        gptq_model_name="test-model-gptq",
        evaluation_texts=["one", "two"],
        evaluation_text_source="fixture",
    )
    original_logits = [
        np.array([[1.0, 0.0, 3.0, 2.0, 4.0, 5.0]], dtype=np.float32)
    ]
    gptq_logits = [
        np.array([[1.0, 0.5, 2.5, 2.0, 4.0, 5.5]], dtype=np.float32)
    ]

    record = gptq._make_logit_record(
        config=config,
        original_logits=original_logits,
        original_loss=2.0,
        gptq_logits=gptq_logits,
        gptq_loss=2.5,
        method_elapsed_seconds=12.5,
        method_cuda_peak_allocated_mb=123.0,
        method_cuda_peak_reserved_mb=234.0,
        total_input_tokens=42,
        method_tokens_per_second=3.36,
        method_ms_per_token=297.619,
    )

    assert record.model_name == "test-model"
    assert record.scope == "external_baseline"
    assert record.method == "external_gptq_w4"
    assert record.bitwidth == 4
    assert record.calibration_text_source == "fixture"
    assert record.calibration_text_count == 2
    assert record.logit_mse == pytest.approx(0.125)
    assert record.loss_delta == pytest.approx(0.5)
    assert record.perplexity_ratio == pytest.approx(np.exp(0.5))
    assert 0.0 <= record.top5_token_overlap <= 1.0
    assert record.method_elapsed_seconds == pytest.approx(12.5)
    assert record.method_cuda_peak_allocated_mb == pytest.approx(123.0)
    assert record.method_cuda_peak_reserved_mb == pytest.approx(234.0)
    assert record.total_input_tokens == 42
    assert record.method_tokens_per_second == pytest.approx(3.36)
    assert record.method_ms_per_token == pytest.approx(297.619)


def test_write_logit_csv_writes_expected_fields(tmp_path):
    config = gptq.GptqBaselineConfig()
    record = gptq._make_logit_record(
        config=config,
        original_logits=[np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)],
        original_loss=1.0,
        gptq_logits=[np.array([[0.0, 1.1, 2.0, 2.9, 4.0]], dtype=np.float32)],
        gptq_loss=1.1,
    )
    path = tmp_path / "metrics.csv"

    gptq._write_logit_csv(path, [record])

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["method"] == "external_gptq_w4"
    assert rows[0]["bitwidth"] == "4"
    assert "method_elapsed_seconds" in rows[0]
    assert "method_cuda_peak_allocated_mb" in rows[0]
    assert "method_cuda_peak_reserved_mb" in rows[0]
    assert "total_input_tokens" in rows[0]
    assert "method_tokens_per_second" in rows[0]
    assert "method_ms_per_token" in rows[0]
    assert "reference_weight_bytes" in rows[0]
    assert "estimated_total_artifact_bytes" in rows[0]


def test_baseline_requires_cuda_before_loading_models(monkeypatch):
    monkeypatch.setattr(
        gptq,
        "_resolve_torch_device",
        lambda device: torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="requires CUDA"):
        gptq.run_gptq_baseline(gptq.GptqBaselineConfig())
