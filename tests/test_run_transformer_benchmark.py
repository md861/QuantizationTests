"""Tests for the safer Milestone 3 transformer benchmark runner."""

from __future__ import annotations

import argparse

import pytest

from experiments import run_transformer_benchmark as runner


def _args(**overrides):
    defaults = dict(
        preset="pythia-14m-int8-baseline",
        local_files_only=False,
        results_dir=None,
        plots_dir=None,
        save_plots=False,
        delete_hf_cache_after=False,
        no_incremental_results=False,
        eval_text_file=None,
        max_eval_texts=None,
        device="auto",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_pythia_int8_preset_builds_conservative_config():
    config = runner.build_config(_args())

    assert config.model_name == "EleutherAI/pythia-14m"
    assert config.single_layer_name is None
    assert config.bitwidths == [8]
    assert config.top_width_pair_fractions == []
    assert config.save_plots is False
    assert config.delete_hf_cache_after is False
    assert config.incremental_results is True
    assert config.calibration_text_source == "built-in calibration texts"


def test_tinyllama_smoke_preset_uses_single_layer_and_one_text():
    config = runner.build_config(_args(preset="tinyllama-1.1b-int4-smoke"))

    assert config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert config.single_layer_name == "model.layers.0.self_attn.q_proj"
    assert config.bitwidths == [4]
    assert config.top_width_pair_fractions == []
    assert config.calibration_texts == ["Quantization smoke test."]
    assert "tinyllama" in str(config.results_dir)


def test_config_honors_local_files_and_overrides(tmp_path):
    config = runner.build_config(
        _args(
            local_files_only=True,
            results_dir=tmp_path / "results",
            plots_dir=tmp_path / "plots",
            save_plots=True,
            delete_hf_cache_after=True,
            no_incremental_results=True,
            device="cpu",
        )
    )

    assert config.local_files_only is True
    assert config.results_dir == tmp_path / "results"
    assert config.plots_dir == tmp_path / "plots"
    assert config.save_plots is True
    assert config.delete_hf_cache_after is True
    assert config.incremental_results is False
    assert config.device == "cpu"


def test_config_loads_eval_text_file(tmp_path):
    text_path = tmp_path / "eval.txt"
    text_path.write_text("One text.\n\nTwo text.\n", encoding="utf-8")

    config = runner.build_config(
        _args(eval_text_file=text_path, max_eval_texts=1)
    )

    assert config.calibration_texts == ["One text."]
    assert config.calibration_text_source == str(text_path)


def test_configure_threads_validates_positive_count():
    with pytest.raises(ValueError, match="torch-threads"):
        runner.configure_threads(0)


def test_metadata_writer_records_numeric_elapsed_and_counts(tmp_path, monkeypatch):
    args = _args(preset="tinyllama-1.1b-int4-smoke", device="cpu")
    config = runner.build_config(args)
    monkeypatch.setattr(runner, "_git_commit_hash", lambda: "abc123")

    metadata = runner._collect_benchmark_metadata(
        args=args,
        config=config,
        elapsed_seconds=1.23456,
        counts={"weight": 1, "activation": 1, "logit": 1},
    )
    path = runner._write_benchmark_metadata(tmp_path, metadata)

    assert path.name == "benchmark_metadata.json"
    assert metadata["elapsed_seconds"] == 1.235
    assert metadata["commit_hash"] == "abc123"
    assert metadata["counts"] == {"weight": 1, "activation": 1, "logit": 1}
    assert metadata["resolved_device"] == "cpu"
    assert path.read_text(encoding="utf-8").strip().startswith("{")


def test_download_only_uses_local_files_flag(monkeypatch):
    calls = []

    def fake_download(model_name, local_files_only):
        calls.append((model_name, local_files_only))

    monkeypatch.setattr(runner, "download_artifacts", fake_download)
    monkeypatch.setattr(runner, "configure_threads", lambda torch_threads: None)

    exit_code = runner.main(
        [
            "pythia-14m-int8-baseline",
            "--download-only",
            "--local-files-only",
        ]
    )

    assert exit_code == 0
    assert calls == [("EleutherAI/pythia-14m", True)]


def test_list_presets_exits_without_running(monkeypatch, capsys):
    monkeypatch.setattr(
        runner,
        "build_config",
        lambda config: pytest.fail("should not run benchmark"),
    )

    exit_code = runner.main(["--list-presets"])

    assert exit_code == 0
    assert "pythia-14m-int8-baseline" in capsys.readouterr().out
