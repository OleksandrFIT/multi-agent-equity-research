import json
from pathlib import Path

FIX = Path(__file__).parent / "fixtures" / "golden" / "retrieval_golden.json"


def test_golden_corpus_is_self_consistent():
    data = json.loads(FIX.read_text())
    doc_ids = {d["id"] for d in data["documents"]}
    tickers_with_docs = {d["ticker"] for d in data["documents"]}
    assert len(doc_ids) == len(data["documents"]), "duplicate document ids"
    assert len(data["cases"]) >= 12
    for case in data["cases"]:
        exp = case["expected"]
        exp = exp if isinstance(exp, list) else [exp]
        assert exp, f"empty expected for {case['question']}"
        for e in exp:
            assert e in doc_ids, f"expected id {e} not in documents"
        assert case["ticker"] in tickers_with_docs
