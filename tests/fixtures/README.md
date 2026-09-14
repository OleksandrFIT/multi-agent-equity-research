# Golden fixtures — ground-truth method

Each fixture is a point-in-time snapshot. Do NOT auto-refresh; add new dated
files for new snapshots.

- `aapl_facts.json` — facts captured from Apple's cited 10-K (record `source_filing`
  and `filed_at`). `expected` values are hand-computed from those facts AND
  cross-checked against one independent reference (macrotrends or stockanalysis);
  record the reference URL in `reference`.
- `edge_cases.json` — synthetic/real cases covering: loss-making company (pe = nan),
  short history, off-cycle fiscal year, missing filing.

> NOTE: the numeric facts in `aapl_facts.json` still require live verification
> against the cited SEC filing and the reference URL before being trusted in
> production. They are currently self-consistent (expected == computed) so the
> unit test is a true regression guard, but the ground-truth cross-check is a
> manual, network-requiring follow-up.
