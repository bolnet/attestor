"""Hard invariant — configs/attestor.yaml and configs/bench.yaml must
have disjoint dotted-key sets.

If a key like ``models:`` or ``embedder:`` appears in both files, we
cannot tell which file "wins" without picking an arbitrary order, and
benchmark vs production drift becomes silent. This test fires at
commit time so a duplicate is caught before it lands.

Companion runtime check lives in
``attestor.bench_config.get_bench()`` which raises
:class:`KeyOverlapError` on the same condition.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from attestor.bench_config import _flatten_keys, assert_disjoint_keys, KeyOverlapError


REPO_ROOT = Path(__file__).resolve().parents[1]
ATTESTOR_YAML = REPO_ROOT / "configs" / "attestor.yaml"
BENCH_YAML = REPO_ROOT / "configs" / "bench.yaml"


@pytest.mark.unit
def test_attestor_yaml_exists():
    assert ATTESTOR_YAML.exists(), f"missing: {ATTESTOR_YAML}"


@pytest.mark.unit
def test_bench_yaml_exists():
    assert BENCH_YAML.exists(), f"missing: {BENCH_YAML}"


@pytest.mark.unit
def test_attestor_and_bench_yaml_have_disjoint_keys():
    """The two YAMLs must not share any dotted key.

    On failure, the assertion message lists the duplicates so the
    operator sees exactly what to delete from bench.yaml.
    """
    attestor = yaml.safe_load(ATTESTOR_YAML.read_text()) or {}
    bench = yaml.safe_load(BENCH_YAML.read_text()) or {}

    overlap = _flatten_keys(attestor) & _flatten_keys(bench)
    assert not overlap, (
        f"configs/attestor.yaml and configs/bench.yaml share "
        f"{len(overlap)} key(s): {sorted(overlap)}.\n"
        f"  Bench file is strictly bench-only knobs (variants, "
        f"categories, target scores, output paths). Stack knobs live "
        f"in attestor.yaml only — remove the duplicates from bench.yaml."
    )


@pytest.mark.unit
def test_assert_disjoint_keys_raises_on_overlap():
    """Sanity check on the helper itself — ensures a fabricated overlap
    actually triggers the error."""
    a = {"models": {"answerer": "x"}, "stack": {"foo": 1}}
    b = {"models": {"answerer": "y"}, "bench": {"variant": "s"}}
    with pytest.raises(KeyOverlapError) as exc:
        assert_disjoint_keys(a, b)
    assert "models.answerer" in str(exc.value)


@pytest.mark.unit
def test_assert_disjoint_keys_passes_on_clean_split():
    """The actual files should pass the helper too."""
    a = {"stack": {"models": {"answerer": "x"}}, "image": {"ref": "r"}}
    b = {"bench": {"lme": {"variant": "s"}}}
    assert_disjoint_keys(a, b)  # must not raise


@pytest.mark.unit
def test_flatten_keys_handles_nested_and_lists():
    """Lists are leaves — their items are not flattened."""
    node = {
        "a": 1,
        "b": {"c": 2, "d": [1, 2, 3]},
        "e": {"f": {"g": 3}},
    }
    keys = _flatten_keys(node)
    assert keys == {"a", "b.c", "b.d", "e.f.g"}
