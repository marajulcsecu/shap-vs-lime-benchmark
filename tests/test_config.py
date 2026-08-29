from xaibench import config


def test_cells_are_18_unique_pairs():
    assert len(config.CELLS) == 18
    assert len(set(config.CELLS)) == 18
    assert set(d for d, _ in config.CELLS) == set(config.DATASETS)
    assert set(m for _, m in config.CELLS) == set(config.MODELS)


def test_quick_is_cheaper_than_full_on_every_budget():
    q, f = config.QUICK, config.FULL
    assert q.n_explain < f.n_explain
    assert q.rerun_repeats < f.rerun_repeats
    assert q.robust_neighbours < f.robust_neighbours
    assert q.bootstrap_refits < f.bootstrap_refits
    assert q.ci_resamples < f.ci_resamples
    assert max(q.lime_num_samples) < max(f.lime_num_samples)


def test_default_profile_is_quick_so_accidents_are_cheap(monkeypatch):
    monkeypatch.delenv("XAIB_PROFILE", raising=False)
    assert config.get_profile().name == "QUICK"
    monkeypatch.setenv("XAIB_PROFILE", "FULL")
    assert config.get_profile().name == "FULL"


def test_protocol_constants_match_frozen_protocol():
    # These values are normative in docs/PROTOCOL.md. If a test here fails,
    # either the code drifted or the protocol was amended without logging it.
    assert config.SEED == 42
    assert config.ALPHA == 0.05
    assert config.RRF_K == 60
    assert config.AOPC_MAX_K == 10
    assert config.CALIB_FRACTION == 0.30
    assert config.FULL.rerun_repeats == 10
    assert config.FULL.robust_neighbours == 5
    assert config.FULL.bootstrap_refits == 10
    assert config.FULL.n_explain == 150
    assert config.FULL.ci_resamples == 2000
    assert config.FULL.lime_num_samples == (5000, 1000)
    assert config.ENSEMBLE_MODELS == ("rf", "xgb", "lgbm")
