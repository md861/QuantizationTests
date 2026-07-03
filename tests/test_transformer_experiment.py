"""Tests for the Milestone 3 transformer quantization harness."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
import torch

from experiments.transformer_experiment import (
    ActivationRecord,
    LogitRecord,
    TransformerConfig,
    WeightRecord,
    _common_method_keys,
    _dequantize_method,
    _effective_top_width_pair_fractions,
    _configured_method_keys,
    _extract_weight,
    _fraction_tag,
    _get_linear_layers,
    load_text_batch,
    _plot_dashboard,
    _resolve_group_sizes,
    _run_weight_experiment,
    _set_weight,
    _top5_overlap,
    run_transformer_experiment,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_config(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("transformer")
    return TransformerConfig(
        model_name="sshleifer/tiny-gpt2",
        calibration_texts=["Hello world."],
        single_layer_name="transformer.h.0.mlp.c_fc",
        bitwidths=[4, 8],
        row_group_sizes=[4],
        row_group_fractions=[0.5, 0.25, 0.0625],
        top_width_pair_fractions=[0.05, 0.10, 0.20],
        results_dir=tmp / "results",
        plots_dir=tmp / "plots",
        save_plots=True,
        delete_hf_cache_after=False,
    )


@pytest.fixture(scope="module")
def experiment_results(tiny_config):
    return run_transformer_experiment(tiny_config)


# ── unit: weight / group utilities ───────────────────────────────────────────


def test_extract_set_weight_roundtrip_linear():
    layer = torch.nn.Linear(8, 16, bias=False)
    original = _extract_weight(layer)
    assert original.shape == (8, 16)  # (in, out) layout
    _set_weight(layer, original)
    np.testing.assert_allclose(original, _extract_weight(layer), atol=1e-6)


def test_extract_set_weight_modifies_parameter():
    layer = torch.nn.Linear(4, 8, bias=False)
    w = _extract_weight(layer)
    _set_weight(layer, w * 2.0)
    np.testing.assert_allclose(_extract_weight(layer), w * 2.0, atol=1e-6)


def test_resolve_group_sizes_deduplicates():
    sizes = _resolve_group_sizes([4], [0.5, 0.25, 0.0625], n_rows=8)
    assert len(sizes) == len(set(sizes)), "group sizes should be unique"
    assert 4 in sizes


def test_resolve_group_sizes_fixed_preserved():
    sizes = _resolve_group_sizes([4], [], n_rows=16)
    assert sizes == [4]


def test_resolve_group_sizes_fraction_at_least_one():
    sizes = _resolve_group_sizes([], [0.5, 0.25, 0.0625], n_rows=2)
    assert all(g >= 1 for g in sizes)


def test_resolve_group_sizes_order_fixed_first():
    sizes = _resolve_group_sizes([4], [0.5], n_rows=8)
    assert sizes[0] == 4  # fixed sizes come first


def test_fraction_tag_integer_percent():
    assert _fraction_tag(0.05) == "p5"
    assert _fraction_tag(0.10) == "p10"
    assert _fraction_tag(0.20) == "p20"


def test_effective_top_width_pair_fractions_cap_and_dedupe():
    fractions = _effective_top_width_pair_fractions(
        [0.05, 0.10, 0.20],
        max_total_pairs=32640,
        max_rotation_pairs=1000,
    )
    expected_cap = 1000 / 32640
    assert fractions == [expected_cap]


def test_effective_top_width_pair_fractions_preserves_uncapped_values():
    fractions = _effective_top_width_pair_fractions(
        [0.05, 0.10],
        max_total_pairs=2016,
        max_rotation_pairs=1000,
    )
    assert fractions == [0.05, 0.10]


def test_top5_overlap_identical():
    logits = [np.random.default_rng(0).standard_normal((5, 100))]
    assert _top5_overlap(logits, logits) == pytest.approx(1.0)


def test_top5_overlap_range():
    rng = np.random.default_rng(42)
    orig = [rng.standard_normal((5, 100))]
    q = [rng.standard_normal((5, 100))]
    assert 0.0 <= _top5_overlap(orig, q) <= 1.0


def test_dashboard_loss_delta_uses_symmetric_log_scale():
    weight_records = [
        WeightRecord(
            "m", "l", "(2, 2)", "global", 4, 0, 0.0, 0.0,
            1.0, 0.1, 0.9, 1.0, 0.0, 0.0,
        )
    ]
    activation_records = [ActivationRecord("m", "l", "global", 4, 1.0, 0.9, 0.1)]
    logit_records = [
        LogitRecord(
            "m", "all_layers", "global", 4, "fixture", 2, 1.0, 0.9, 0.5,
            2.0, 1.0, 1.0, math.e**2, math.e, math.e,
        ),
        LogitRecord(
            "m", "all_layers", "global", 8, "fixture", 2, 1e-4, 0.99, 1.0,
            0.999999, 1.0, -1e-6, math.exp(0.999999), math.e,
            math.exp(-1e-6),
        ),
    ]

    fig = _plot_dashboard(weight_records, activation_records, logit_records)
    try:
        assert fig.axes[3].get_xscale() == "symlog"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


# ── unit: weight experiment ───────────────────────────────────────────────────


def _make_config(**kwargs) -> TransformerConfig:
    defaults = dict(
        row_group_sizes=[4],
        row_group_fractions=[0.5],
        top_width_pair_fractions=[0.10],
        bitwidths=[4, 8],
        save_plots=False,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def test_weight_experiment_has_both_bitwidths():
    config = _make_config(bitwidths=[4, 8])
    weight = np.random.default_rng(7).standard_normal((8, 16)).astype(np.float32)
    records, method_deqs = _run_weight_experiment(config, "test.layer", weight)
    bws = {r.bitwidth for r in records}
    assert bws == {4, 8}
    assert all(isinstance(k[1], int) for k in method_deqs)


def test_weight_experiment_int4_only():
    config = _make_config(bitwidths=[4])
    weight = np.random.default_rng(1).standard_normal((8, 16)).astype(np.float32)
    records, _ = _run_weight_experiment(config, "l", weight)
    assert all(r.bitwidth == 4 for r in records)


def test_weight_experiment_int8_only():
    config = _make_config(bitwidths=[8])
    weight = np.random.default_rng(2).standard_normal((8, 16)).astype(np.float32)
    records, _ = _run_weight_experiment(config, "l", weight)
    assert all(r.bitwidth == 8 for r in records)


def test_weight_experiment_dynamic_group_sizes_present():
    config = _make_config(
        row_group_sizes=[],
        row_group_fractions=[0.5, 0.25],
        bitwidths=[4],
    )
    weight = np.random.default_rng(3).standard_normal((8, 16)).astype(np.float32)
    records, _ = _run_weight_experiment(config, "l", weight)
    row_methods = {r.method for r in records if "row_grouped" in r.method}
    # n_rows=8: g = round(8*0.5)=4, round(8*0.25)=2 → two distinct sizes
    assert "row_grouped_g4" in row_methods
    assert "row_grouped_g2" in row_methods


def test_weight_experiment_top_width_fractions():
    config = _make_config(
        row_group_fractions=[],
        top_width_pair_fractions=[0.05, 0.10, 0.20],
        bitwidths=[4],
    )
    weight = np.random.default_rng(5).standard_normal((8, 16)).astype(np.float32)
    records, _ = _run_weight_experiment(config, "l", weight)
    tags = {r.method for r in records if "top_width" in r.method}
    assert any("p5" in t for t in tags)
    assert any("p10" in t for t in tags)
    assert any("p20" in t for t in tags)


def test_weight_experiment_dequantized_shape():
    config = _make_config()
    weight = np.random.default_rng(3).standard_normal((8, 16)).astype(np.float32)
    _, method_deqs = _run_weight_experiment(config, "l", weight)
    for key, deq in method_deqs.items():
        assert deq.shape == weight.shape, f"{key}: shape mismatch"


def test_dequantize_method_matches_weight_experiment_outputs():
    config = _make_config(
        row_group_sizes=[4],
        row_group_fractions=[0.5],
        top_width_pair_fractions=[0.10],
        bitwidths=[4, 8],
    )
    weight = np.random.default_rng(11).standard_normal((8, 16)).astype(np.float32)
    _, method_deqs = _run_weight_experiment(config, "l", weight)

    for (method, bitwidth), expected in method_deqs.items():
        actual = _dequantize_method(
            config,
            weight,
            method,
            bitwidth,
            top_width_pair_fractions=config.top_width_pair_fractions,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_common_method_keys_intersects_layer_specific_group_sizes():
    all_method_deqs = {
        "small": {
            ("global", 4): np.zeros((4, 4), dtype=np.float32),
            ("row_grouped_g4", 4): np.zeros((4, 4), dtype=np.float32),
            ("row_grouped_g2", 4): np.zeros((4, 4), dtype=np.float32),
            ("global", 8): np.zeros((4, 4), dtype=np.float32),
        },
        "large": {
            ("global", 4): np.zeros((8, 4), dtype=np.float32),
            ("row_grouped_g4", 4): np.zeros((8, 4), dtype=np.float32),
            ("row_grouped_g8", 4): np.zeros((8, 4), dtype=np.float32),
            ("global", 8): np.zeros((8, 4), dtype=np.float32),
        },
    }

    assert _common_method_keys(all_method_deqs) == [
        ("global", 4),
        ("row_grouped_g4", 4),
        ("global", 8),
    ]


def test_weight_experiment_metrics_finite():
    config = _make_config()
    weight = np.random.default_rng(5).standard_normal((16, 32)).astype(np.float32)
    records, _ = _run_weight_experiment(config, "l", weight)
    for r in records:
        assert np.isfinite(r.mse), f"{r.method} bw={r.bitwidth}: MSE not finite"
        assert np.isfinite(r.snr_db), f"{r.method} bw={r.bitwidth}: SNR not finite"


def test_weight_experiment_int8_mse_le_int4():
    config = _make_config(row_group_fractions=[], top_width_pair_fractions=[])
    weight = np.random.default_rng(9).standard_normal((16, 32)).astype(np.float32)
    records, _ = _run_weight_experiment(config, "l", weight)
    mse4 = next(r.mse for r in records if r.method == "global" and r.bitwidth == 4)
    mse8 = next(r.mse for r in records if r.method == "global" and r.bitwidth == 8)
    assert mse8 <= mse4, "INT8 global should have lower MSE than INT4 global"


def test_weight_experiment_rotation_metadata_non_rotation_zero():
    config = _make_config(top_width_pair_fractions=[0.10], bitwidths=[4])
    weight = np.random.default_rng(9).standard_normal((8, 16)).astype(np.float32)
    records, _ = _run_weight_experiment(config, "l", weight)
    for r in records:
        if "rotate" not in r.method:
            assert r.rotation_count == 0
            assert r.rotation_candidate_fraction == 0.0


def test_get_linear_layers_excludes_lm_head():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    layers = _get_linear_layers(model)
    assert "lm_head" not in layers
    assert len(layers) > 0


def test_load_text_batch_ignores_comments_and_splits_paragraphs(tmp_path):
    path = tmp_path / "texts.txt"
    path.write_text(
        "# metadata\n"
        "First paragraph line one.\n"
        "First paragraph line two.\n"
        "\n"
        "# another comment\n"
        "Second paragraph.\n",
        encoding="utf-8",
    )

    assert load_text_batch(path) == [
        "First paragraph line one. First paragraph line two.",
        "Second paragraph.",
    ]


def test_load_text_batch_honors_max_texts(tmp_path):
    path = tmp_path / "texts.txt"
    path.write_text("One.\n\nTwo.\n\nThree.\n", encoding="utf-8")

    assert load_text_batch(path, max_texts=2) == ["One.", "Two."]


def test_load_text_batch_rejects_empty_resource(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("# comments only\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No evaluation texts"):
        load_text_batch(path)


# ── integration tests ─────────────────────────────────────────────────────────


def test_experiment_returns_three_record_lists(experiment_results):
    wr, ar, lr = experiment_results
    assert len(wr) > 0
    assert len(ar) > 0
    assert len(lr) > 0


def test_weight_records_contain_both_bitwidths(experiment_results):
    wr, _, _ = experiment_results
    assert {r.bitwidth for r in wr} == {4, 8}


def test_activation_records_contain_both_bitwidths(experiment_results):
    _, ar, _ = experiment_results
    assert {r.bitwidth for r in ar} == {4, 8}


def test_logit_records_contain_both_bitwidths(experiment_results):
    _, _, lr = experiment_results
    assert {r.bitwidth for r in lr} == {4, 8}


def test_weight_records_fields_valid(experiment_results):
    wr, _, _ = experiment_results
    for r in wr:
        assert isinstance(r, WeightRecord)
        assert r.model_name == "sshleifer/tiny-gpt2"
        assert np.isfinite(r.mse)
        assert np.isfinite(r.cosine_similarity)
        assert 0.0 <= r.zero_fraction <= 1.0
        assert r.bitwidth in (4, 8)


def test_activation_records_fields_valid(experiment_results):
    _, ar, _ = experiment_results
    for r in ar:
        assert isinstance(r, ActivationRecord)
        assert np.isfinite(r.activation_mse)
        assert r.activation_mse >= 0.0
        assert np.isfinite(r.activation_cosine_similarity)
        assert r.bitwidth in (4, 8)


def test_logit_records_fields_valid(experiment_results):
    _, _, lr = experiment_results
    for r in lr:
        assert isinstance(r, LogitRecord)
        assert np.isfinite(r.logit_mse)
        assert np.isfinite(r.loss)
        assert r.loss_delta == pytest.approx(r.loss - r.original_loss, abs=1e-5)
        assert r.perplexity == pytest.approx(math.exp(r.loss), rel=1e-6)
        assert r.original_perplexity == pytest.approx(
            math.exp(r.original_loss), rel=1e-6
        )
        assert r.perplexity_ratio == pytest.approx(
            r.perplexity / r.original_perplexity, rel=1e-6
        )
        assert 0.0 <= r.top5_token_overlap <= 1.0
        assert r.bitwidth in (4, 8)
        assert r.calibration_text_source == "built-in calibration texts"
        assert r.calibration_text_count == 1


def test_method_names_consistent_across_all_records(experiment_results):
    wr, ar, lr = experiment_results
    weight_keys = {(r.method, r.bitwidth) for r in wr}
    activation_keys = {(r.method, r.bitwidth) for r in ar}
    logit_keys = {(r.method, r.bitwidth) for r in lr}
    assert weight_keys == activation_keys
    assert logit_keys == weight_keys


def test_csv_files_written(tiny_config, experiment_results):
    assert (tiny_config.results_dir / "transformer_weight_metrics.csv").exists()
    assert (tiny_config.results_dir / "transformer_activation_metrics.csv").exists()
    assert (tiny_config.results_dir / "transformer_logit_metrics.csv").exists()


def test_dashboard_files_written(tiny_config, experiment_results):
    assert (tiny_config.plots_dir / "transformer_dashboard.png").exists()


def test_logit_csv_includes_perplexity_columns(tiny_config, experiment_results):
    path = tiny_config.results_dir / "transformer_logit_metrics.csv"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        assert "perplexity" in reader.fieldnames
        assert "original_perplexity" in reader.fieldnames
        assert "perplexity_ratio" in reader.fieldnames
        assert "calibration_text_source" in reader.fieldnames
        assert "calibration_text_count" in reader.fieldnames


def test_int8_activation_drift_le_int4(experiment_results):
    _, ar, _ = experiment_results
    mse4 = np.mean([r.activation_mse for r in ar if r.method == "global" and r.bitwidth == 4])
    mse8 = np.mean([r.activation_mse for r in ar if r.method == "global" and r.bitwidth == 8])
    assert mse8 <= mse4 + 1e-7, "INT8 global activation drift should not exceed INT4"


def test_int8_logit_drift_le_int4(experiment_results):
    _, _, lr = experiment_results
    mse4 = next(r.logit_mse for r in lr if r.method == "global" and r.bitwidth == 4)
    mse8 = next(r.logit_mse for r in lr if r.method == "global" and r.bitwidth == 8)
    assert mse8 <= mse4 + 1e-7, "INT8 global logit drift should not exceed INT4"


def test_configured_method_keys_filters_logit_methods():
    class DummyLinear(torch.nn.Linear):
        pass

    layers = {"a": DummyLinear(4, 4, bias=False)}
    config = TransformerConfig(
        bitwidths=[4],
        row_group_sizes=[4, 8],
        row_group_fractions=[],
        top_width_pair_fractions=[],
        logit_only=True,
        logit_method_names=["global", "row_grouped_g4"],
    )

    assert _configured_method_keys(config, layers) == [
        ("global", 4),
        ("row_grouped_g4", 4),
    ]
