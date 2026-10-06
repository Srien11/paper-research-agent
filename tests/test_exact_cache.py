from paper_research_agent.exact_cache import ExactCache, exact_key


def test_absolute_ttl_lru_and_disabled_cache() -> None:
    now = [0.0]
    cache = ExactCache[int](2, 5, clock=lambda: now[0])
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert cache.get("b") is None
    now[0] = 4
    assert cache.get("a") == 1
    now[0] = 5
    assert cache.get("a") is None  # A hit does not extend the deadline.
    disabled = ExactCache[int](0)
    disabled.put("a", 1)
    assert disabled.get("a") is None


def test_keys_are_exact_order_sensitive_and_do_not_retain_inputs() -> None:
    assert exact_key(["q", ["a", "b"]]) != exact_key(["q", ["b", "a"]])
    assert exact_key("q ") != exact_key("q")
    assert len(exact_key("private question")) == 64


def test_environment_rollback(monkeypatch) -> None:
    monkeypatch.setenv("PRA_EXACT_CACHE_ENABLED", "false")
    cache = ExactCache[int]()
    cache.put("a", 1)
    assert cache.get("a") is None
