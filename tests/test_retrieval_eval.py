from equity_research.eval.retrieval_eval import hit_at_k


def test_hit_at_k_counts_expected_in_topk():
    def retrieve_fn(ticker, question):
        return {"q1": ["d1", "d2", "d3"], "q2": ["dx", "dy", "dz"]}[question]

    cases = [
        {"ticker": "AAA", "question": "q1", "expected": "d2"},
        {"ticker": "AAA", "question": "q2", "expected": "d9"},
    ]
    assert hit_at_k(retrieve_fn, cases, k=3) == 0.5
