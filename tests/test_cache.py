from src.inference.cache import ResponseCache


def test_response_cache_hit_and_miss():
    cache = ResponseCache(ttl_seconds=60, max_entries=5)

    assert cache.get("câu test", "finetune") is None

    cache.set("câu test", {"corrected": "Câu test."}, "finetune")
    val = cache.get("câu test", "finetune")
    assert val is not None
    assert val["corrected"] == "Câu test."


def test_response_cache_eviction():
    cache = ResponseCache(ttl_seconds=60, max_entries=2)

    cache.set("text1", "val1")
    cache.set("text2", "val2")
    cache.set("text3", "val3")

    # text1 should be evicted as max_entries is 2
    assert cache.get("text1") is None
    assert cache.get("text2") == "val2"
    assert cache.get("text3") == "val3"
