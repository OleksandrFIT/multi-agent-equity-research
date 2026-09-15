from equity_research.eval.retrieval_eval import hit_at_k


def test_hit_at_k_counts_expected_in_topk():
    def retrieve_fn(ticker, question):
        return {"q1": ["d1", "d2", "d3"], "q2": ["dx", "dy", "dz"]}[question]

    cases = [
        {"ticker": "AAA", "question": "q1", "expected": "d2"},
        {"ticker": "AAA", "question": "q2", "expected": "d9"},
    ]
    assert hit_at_k(retrieve_fn, cases, k=3) == 0.5


def test_hit_at_k_accepts_list_expected():
    from equity_research.eval.retrieval_eval import hit_at_k

    def retrieve_fn(ticker, question):
        return {"q1": ["d1", "d2", "d3"]}[question]

    cases = [{"ticker": "AAA", "question": "q1", "expected": ["d9", "d2"]}]  # any expected in top-k
    assert hit_at_k(retrieve_fn, cases, k=3) == 1.0


def test_mrr_uses_first_relevant_rank():
    from equity_research.eval.retrieval_eval import mrr

    def retrieve_fn(ticker, question):
        return {"q1": ["d1", "d2", "d3"], "q2": ["dx", "dy", "dz"]}[question]

    cases = [
        {"ticker": "A", "question": "q1", "expected": "d2"},   # rank 2 -> 1/2
        {"ticker": "A", "question": "q2", "expected": ["dz"]}, # rank 3 -> 1/3
    ]
    assert abs(mrr(retrieve_fn, cases, k=3) - ((0.5 + (1 / 3)) / 2)) < 1e-9


def test_mrr_zero_when_not_found():
    from equity_research.eval.retrieval_eval import mrr

    cases = [{"ticker": "A", "question": "q", "expected": "zzz"}]
    assert mrr(lambda t, q: ["a", "b"], cases, k=2) == 0.0
