from agent_memory.store.registry import BACKEND_REGISTRY, discover_backends


def test_builtins_present_after_discover():
    discover_backends()
    for name in ("sqlite", "chroma", "networkx", "arangodb", "postgres"):
        assert name in BACKEND_REGISTRY, f"{name} should be discovered"


def test_discover_is_idempotent():
    discover_backends()
    snapshot = dict(BACKEND_REGISTRY)
    discover_backends()
    assert BACKEND_REGISTRY == snapshot
