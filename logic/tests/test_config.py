"""Tests for logic.config."""
import json

import pytest

from logic.config import Config, config_from_dict, config_to_dict, load_config, save_config


def test_json_matches_defaults() -> None:
    assert load_config() == Config()


def test_partial_override_keeps_other_defaults() -> None:
    cfg = config_from_dict({"fall": {"V_DROP": 2.0}, "filters": {"one_euro": {"beta": 0.5}}})
    assert cfg.fall.V_DROP == 2.0
    assert cfg.fall.PHI_DOWN == Config().fall.PHI_DOWN
    assert cfg.filters.one_euro.beta == 0.5
    assert cfg.filters.one_euro.min_cutoff == Config().filters.one_euro.min_cutoff


def test_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="fall.V_DORP"):
        config_from_dict({"fall": {"V_DORP": 2.0}})


def test_roundtrip(tmp_path) -> None:
    cfg = config_from_dict({"eval": {"grid_T_HOLD": [3.0]}, "features": {"y_floor_override": 700.0}})
    path = tmp_path / "c.json"
    save_config(cfg, path)
    assert load_config(path) == cfg
    assert json.loads(path.read_text())["eval"]["grid_T_HOLD"] == [3.0]
    assert config_to_dict(cfg)["features"]["y_floor_override"] == 700.0
