import pytest

from unfakeable.config import ConfigError, load_config


def test_defaults_and_overrides(tmp_path):
    cfg = load_config(None, ["seed=7", "preproc.n_frames=8", "fusion.n_components=null",
                             "tuning.classifiers.random_forest.n_iter=3"])
    assert cfg.seed == 7 and cfg.preproc.n_frames == 8 and cfg.fusion.n_components is None
    assert cfg.tuning.classifiers["random_forest"]["n_iter"] == 3
    assert "space" in cfg.tuning.classifiers["random_forest"]  # deep merge keeps siblings


def test_yaml_merge_and_unknown_key(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("run_name: x\ndata:\n  test_size: 0.25\n")
    cfg = load_config(str(f))
    assert cfg.run_name == "x" and cfg.data.test_size == 0.25 and cfg.data.balance is True
    f.write_text("data:\n  not_a_key: 1\n")
    with pytest.raises(ConfigError):
        load_config(str(f))


def test_fingerprint_changes_with_preproc():
    a, b = load_config(None, []), load_config(None, ["preproc.n_frames=8"])
    assert a.prep_fingerprint() != b.prep_fingerprint()


def test_shipped_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for f in sorted(root.glob("*.yaml")):
        load_config(str(f))
