from datetime import date

import pytest

from equity_research.data.edgar import NoFilingError, select_filing_asof


class _Filing:
    def __init__(self, filing_date):
        self.filing_date = filing_date


def test_selects_latest_on_or_before_as_of():
    filings = [_Filing(date(2024, 2, 1)), _Filing(date(2024, 5, 1)), _Filing(date(2024, 8, 1))]
    chosen = select_filing_asof(filings, date(2024, 6, 15))
    assert chosen.filing_date == date(2024, 5, 1)


def test_includes_filing_on_exact_as_of():
    filings = [_Filing(date(2024, 5, 1)), _Filing(date(2024, 6, 15))]
    assert select_filing_asof(filings, date(2024, 6, 15)).filing_date == date(2024, 6, 15)


def test_raises_when_no_filing_before_as_of():
    filings = [_Filing(date(2024, 5, 1))]
    with pytest.raises(NoFilingError):
        select_filing_asof(filings, date(2024, 1, 1))
