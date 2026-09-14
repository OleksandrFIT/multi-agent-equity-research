from __future__ import annotations

from typing import Protocol


class FactsSource(Protocol):
    def company_facts(self, ticker: str) -> dict[str, float]: ...


class EdgarProvider:
    """Thin wrapper around edgartools. Normalizes XBRL facts to a flat dict.

    Field paths depend on the installed edgartools version; verify in the
    integration test (later task) and adjust the extraction here if needed.
    """

    def __init__(self, user_agent: str):
        self.user_agent = user_agent

    def company_facts(self, ticker: str) -> dict[str, float]:
        from edgar import Company, set_identity

        set_identity(self.user_agent)
        company = Company(ticker)
        financials = company.financials
        income = financials.income_statement()
        balance = financials.balance_sheet()

        def latest(frame, label: str) -> float:
            row = frame.loc[label]
            return float(row.iloc[0])

        return {
            "net_income": latest(income, "NetIncomeLoss"),
            "revenue": latest(income, "Revenues"),
            "revenue_prev": float(income.loc["Revenues"].iloc[1]),
            "equity": latest(balance, "StockholdersEquity"),
            "total_debt": latest(balance, "LongTermDebt"),
            "eps_ttm": latest(income, "EarningsPerShareDiluted"),
        }
