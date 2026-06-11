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


def test_config_honors_local_files_and_overrides(tmp_path):
    config = runner.build_config(
        _args(
            local_files_only=True,
            results_dir=tmp_path / "results",
            plots_dir=tmp_path / "plots",
            save_plots=True,
            delete_hf_cache_after=True,
            no_incremental_results=True,
        )
    )

    assert config.local_files_only is True
    assert config.results_dir == tmp_path / "results"
    assert config.plots_dir == tmp_path / "plots"
    assert config.save_plots is True
    assert config.delete_hf_cache_after is True
    assert config.incremental_results is False


def test_configure_threads_validates_positive_count():
    with pytest.raises(ValueError, match="torch-threads"):
        runner.configure_threads(0)


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
