"""Tests for the optional bitsandbytes external baseline runner."""

from __future__ import annotations

import argparse
import csv

import numpy as np
import pytest
import torch

from experiments import bitsandbytes_baseline as bnb


def _args(**overrides):
    defaults = dict(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        eval_text_file=None,
        max_eval_texts=None,
        results_dir=None,
        local_files_only=False,
        device="auto",
        compute_dtype="float16",
        double_quant=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_config_uses_eval_text_file(tmp_path):
    text_path = tmp_path / "eval.txt"
    text_path.write_text("First text.\n\nSecond text.\n", encoding="utf-8")

    config = bnb.build_config(
        _args(
            eval_text_file=text_path,
            max_eval_texts=1,
            results_dir=tmp_path / "results",
            local_files_only=True,
            compute_dtype="bfloat16",
            double_quant=True,
        )
    )

    assert config.evaluation_texts == ["First text."]
    assert config.evaluation_text_source == str(text_path)
    assert config.results_dir == tmp_path / "results"
    assert config.local_files_only is True
    assert config.compute_dtype == "bfloat16"
    assert config.double_quant is True


def test_make_bitsandbytes_config_sets_nf4_options():
    calls = {}

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    config = bnb.BitsAndBytesBaselineConfig(
        compute_dtype="bfloat16",
        double_quant=True,
    )

    result = bnb._make_bitsandbytes_config(config, config_cls=FakeBitsAndBytesConfig)

    assert isinstance(result, FakeBitsAndBytesConfig)
    assert calls == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
    }


def test_resolve_compute_dtype_rejects_unknown_value():
    with pytest.raises(ValueError, match="compute dtype"):
        bnb._resolve_compute_dtype("float8")


def test_make_logit_record_computes_shared_metrics():
    config = bnb.BitsAndBytesBaselineConfig(
        model_name="test-model",
        evaluation_texts=["one", "two"],
        evaluation_text_source="fixture",
    )
    original_logits = [
        np.array([[1.0, 0.0, 3.0, 2.0, 4.0, 5.0]], dtype=np.float32)
    ]
    quantized_logits = [
        np.array([[1.0, 0.5, 2.5, 2.0, 4.0, 5.5]], dtype=np.float32)
    ]

    record = bnb._make_logit_record(
        config=config,
        original_logits=original_logits,
        original_loss=2.0,
        quantized_logits=quantized_logits,
        quantized_loss=2.5,
        method_elapsed_seconds=12.5,
        method_cuda_peak_allocated_mb=123.0,
        method_cuda_peak_reserved_mb=234.0,
        total_input_tokens=42,
        method_tokens_per_second=3.36,
        method_ms_per_token=297.619,
    )

    assert record.model_name == "test-model"
    assert record.scope == "external_baseline"
    assert record.method == "external_bitsandbytes_nf4_float16"
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
    config = bnb.BitsAndBytesBaselineConfig()
    record = bnb._make_logit_record(
        config=config,
        original_logits=[np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)],
        original_loss=1.0,
        quantized_logits=[np.array([[0.0, 1.1, 2.0, 2.9, 4.0]], dtype=np.float32)],
        quantized_loss=1.1,
    )
    path = tmp_path / "metrics.csv"

    bnb._write_logit_csv(path, [record])

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["method"] == "external_bitsandbytes_nf4_float16"
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
        bnb,
        "_resolve_torch_device",
        lambda device: torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="requires CUDA"):
        bnb.run_bitsandbytes_baseline(bnb.BitsAndBytesBaselineConfig())
