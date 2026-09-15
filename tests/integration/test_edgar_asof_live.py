from datetime import date

import pytest


@pytest.mark.integration
def test_edgar_asof_returns_pit_facts():
    """Requires network (SEC). Run: uv run pytest -m integration."""
    from equity_research.data.edgar import EdgarProvider

    provider = EdgarProvider(user_agent="Equity Research test@example.com")
    facts_now = provider.company_facts("AAPL", date.today())
    facts_old = provider.company_facts("AAPL", date(2021, 6, 30))
    assert facts_now["revenue"] > 0
    assert facts_old["revenue"] > 0
    # point-in-time: an as_of in 2021 must not see the latest (much larger) revenue
    assert facts_old["revenue"] != facts_now["revenue"]
