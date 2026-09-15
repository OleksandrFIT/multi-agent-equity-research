# Golden set fixtures

Point-in-time reference data for evaluation. Do NOT auto-refresh — add new dated
files for new snapshots so historical baselines stay stable.

## `metrics_golden.json` (deterministic; used by `tests/test_golden_metrics.py`)
Fundamentals facts captured live from EDGAR (latest 10-K on or before
`as_of = 2024-06-14`) for AAPL, MSFT, NVDA, META, JNJ. `price` is the split-adjusted
close on/before that date. `expected` is computed by `compute_fundamental_metrics`
and is self-consistent with `facts`+`price`, so the test is a true regression guard
on the metric math. `reference` points to an independent source for manual cross-check.

Ticker selection note: tickers were chosen to capture cleanly and yield sensible
non-financial ratios. Some tickers (e.g. KO, XOM) currently error in `company_facts`
because edgartools returns a missing metric field, and financials (e.g. JPM) produce
distorted revenue-growth — those are excluded from the golden set rather than
enshrining broken/misleading values.

## `retrieval_golden.json` (deterministic shape test + `-m integration` retrieval test)
Labeled news snippets for AAPL/MSFT/KO across query categories (earnings, legal,
supply, guidance). `cases[].expected` lists the clearly-relevant document id(s).
The integration test runs the real retriever and asserts hit@3 ≥ 0.8.

## `verdicts_golden.json` (`-m integration` regression/stability test)
End-to-end verdict baselines captured from reviewed real runs (seed + temp 0).
These are stability/sanity guards, NOT a claim of market correctness.
