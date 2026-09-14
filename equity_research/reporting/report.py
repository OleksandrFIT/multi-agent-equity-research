from __future__ import annotations

from equity_research.orchestration.aggregator import Verdict


def render_json(verdict: Verdict) -> str:
    return verdict.model_dump_json(indent=2)


def render_markdown(verdict: Verdict) -> str:
    lines = [
        f"# {verdict.ticker} — {verdict.verdict.upper()}",
        f"_as of {verdict.as_of}_  •  score {verdict.score:+.2f}  •  confidence {verdict.confidence:.0%}",
        "",
        verdict.narrative,
        "",
        "## Analyst opinions",
    ]
    for o in verdict.opinions:
        facts = "; ".join(o.key_facts) if o.key_facts else "—"
        lines.append(f"- **{o.agent}** — {o.stance} (score {o.score:+.2f}, conf {o.confidence:.0%}): {o.rationale} _[{facts}]_")
    if verdict.skipped_agents:
        lines.append("")
        lines.append(f"_Skipped agents: {', '.join(verdict.skipped_agents)}_")
    lines += ["", "---", f"> {verdict.disclaimer}"]
    return "\n".join(lines)
