from equity_research.rag.reranker import rerank


def test_rerank_orders_by_scorer_desc():
    items = [("a", {}), ("b", {}), ("c", {})]
    scores = {"a": 0.1, "b": 0.9, "c": 0.5}

    def scorer(query, texts):
        return [scores[t] for t in texts]

    out = rerank("q", items, scorer)
    assert [t for t, _ in out] == ["b", "c", "a"]


def test_rerank_none_scorer_is_identity():
    items = [("a", {}), ("b", {})]
    assert rerank("q", items, None) == items
