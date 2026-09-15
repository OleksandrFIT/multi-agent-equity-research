from __future__ import annotations


def hit_rate_by_class(records: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    buys = [r for r in records if r["verdict"] == "buy"]
    sells = [r for r in records if r["verdict"] == "sell"]
    if buys:
        out["buy"] = sum(1 for r in buys if r["fwd_return"] > 0) / len(buys)
    if sells:
        out["sell"] = sum(1 for r in sells if r["fwd_return"] < 0) / len(sells)
    return out


def mean_return_by_class(records: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in ("buy", "hold", "sell"):
        rs = [r["fwd_return"] for r in records if r["verdict"] == cls]
        if rs:
            out[cls] = sum(rs) / len(rs)
    return out


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def information_coefficient(records: list[dict]) -> float | None:
    if len(records) < 2:
        return None
    ra = _ranks([r["score"] for r in records])
    rb = _ranks([r["fwd_return"] for r in records])
    n = len(records)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def long_short_curve(records: list[dict]) -> list[float]:
    curve: list[float] = []
    cum = 0.0
    for r in records:
        if r["verdict"] == "buy":
            cum += r["fwd_return"]
        elif r["verdict"] == "sell":
            cum -= r["fwd_return"]
        curve.append(cum)
    return curve
