from equity_research.rag.parent_store import ParentStore


def test_put_get_roundtrip(tmp_path):
    ps = ParentStore(tmp_path)
    ps.put("pid1", "long section text")
    assert ps.get("pid1") == "long section text"
    assert ps.has("pid1")
    assert not ps.has("missing")


def test_persists_across_instances(tmp_path):
    ParentStore(tmp_path).put("pid2", "body")
    assert ParentStore(tmp_path).get("pid2") == "body"
