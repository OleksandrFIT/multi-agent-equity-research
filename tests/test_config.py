from pathlib import Path

from equity_research.config import Config


def test_loads_defaults_from_yaml(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "model: qwen2.5:7b\n"
        "temperature: 0.0\n"
        "seed: 7\n"
        "cache_dir: .cache\n"
        "edgar_user_agent: Test test@example.com\n"
        "weights:\n"
        "  fundamentals: 0.4\n"
        "  technical: 0.25\n"
        "  sentiment: 0.15\n"
        "  risk: 0.2\n"
    )
    cfg = Config.load(cfg_file)
    assert cfg.model == "qwen2.5:7b"
    assert cfg.seed == 7
    assert cfg.weights["fundamentals"] == 0.4


def test_normalize_weights_over_available_agents():
    cfg = Config(
        model="m", temperature=0.0, seed=1, cache_dir=".cache",
        edgar_user_agent="x x@x.com",
        weights={"fundamentals": 0.4, "technical": 0.25, "sentiment": 0.15, "risk": 0.2},
    )
    norm = cfg.normalized_weights(["fundamentals", "technical"])
    assert abs(sum(norm.values()) - 1.0) < 1e-9
    assert abs(norm["fundamentals"] - 0.4 / 0.65) < 1e-9


def test_config_has_net_defaults():
    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".cache",
                 edgar_user_agent="x x@x.com",
                 weights={"fundamentals": 0.4, "technical": 0.25, "sentiment": 0.15, "risk": 0.2})
    assert cfg.net["data_timeout"] == 20
    assert cfg.net["data_attempts"] == 3
    assert cfg.net["ollama_timeout"] == 180


def test_config_has_rag_defaults():
    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".cache",
                 edgar_user_agent="x x@x.com",
                 weights={"fundamentals": 0.4, "technical": 0.25, "sentiment": 0.15, "risk": 0.2})
    assert cfg.rag["chroma_dir"] == ".chroma"
    assert cfg.rag["embed_model"] == "nomic-embed-text"
    assert cfg.rag["retrieve_k"] == 6


def test_config_has_backtest_defaults():
    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".cache",
                 edgar_user_agent="x x@x.com",
                 weights={"fundamentals": 0.4, "technical": 0.25, "sentiment": 0.15, "risk": 0.2})
    assert cfg.backtest["horizons"] == [21, 63]
    assert len(cfg.backtest["universe"]) >= 1
    assert cfg.backtest["report_path"].endswith(".md")


def test_role_models_default_to_none():
    from equity_research.config import Config

    cfg = Config(model="qwen2.5:7b", temperature=0.0, seed=1, cache_dir=".c",
                 edgar_user_agent="x", weights={"fundamentals": 1})
    assert cfg.judge_model is None and cfg.narrative_model is None
    assert (cfg.judge_model or cfg.model) == "qwen2.5:7b"


def test_pm_config_defaults_off():
    from equity_research.config import Config

    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".c",
                 edgar_user_agent="x", weights={"fundamentals": 1})
    assert cfg.pm_enabled is False
    assert cfg.pm_weight == 0.6


def test_self_critique_defaults_off():
    from equity_research.config import Config

    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".c",
                 edgar_user_agent="x", weights={"fundamentals": 1})
    assert cfg.self_critique_enabled is False


def test_calibration_config_defaults():
    from equity_research.config import Config

    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".c",
                 edgar_user_agent="x", weights={"fundamentals": 1})
    assert cfg.calibration_enabled is False
    assert cfg.calibration_path == "calibration.json"
    assert cfg.calibration_horizon == 21
