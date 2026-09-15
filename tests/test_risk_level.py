from equity_research.analytics.risk import risk_level
from equity_research.config import Config

CFG = {"vol_low": 0.15, "vol_high": 0.60, "beta_high": 1.5, "drawdown_high": 0.40}


def test_low_everything_is_zero():
    m = {"volatility": 0.15, "beta": 0.0, "max_drawdown": 0.0}
    assert risk_level(m, CFG) == 0.0


def test_high_everything_is_one():
    m = {"volatility": 0.60, "beta": 1.5, "max_drawdown": 0.40}
    assert abs(risk_level(m, CFG) - 1.0) < 1e-9


def test_averages_available_components():
    # vol normalized 1.0, drawdown normalized 0.0, beta absent -> average of [1.0, 0.0] = 0.5
    m = {"volatility": 0.60, "max_drawdown": 0.0}
    assert abs(risk_level(m, CFG) - 0.5) < 1e-9


def test_empty_metrics_is_zero():
    assert risk_level({}, CFG) == 0.0


def test_config_has_risk_defaults():
    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".cache",
                 edgar_user_agent="x x@x.com",
                 weights={"fundamentals": 0.4, "technical": 0.25, "sentiment": 0.15, "risk": 0.2})
    assert cfg.benchmark == "SPY"
    assert cfg.risk["gate_strength"] == 0.5
    assert cfg.risk["vol_high"] == 0.60
