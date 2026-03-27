"""
Tests for compiler.passes.dce

DCE removes ops whose output is never transitively consumed by a live
graph output. These tests cover: explicit outputs, inferred outputs,
full dead chains, nothing-to-remove, and the empty-graph edge case.
"""

from compiler.passes.dce import eliminate_dead_code
from tests.conftest import make_graph, op_names


# --- explicit outputs ---

def test_dead_op_removed_with_explicit_outputs():
    g = make_graph(
        ("mul0", "MUL", ["a", "b"]),
        ("add0", "ADD", ["mul0", "c"]), # never consumed, not a declared output
        outputs=["mul0"],
    )
    eliminate_dead_code(g)

    assert op_names(g) == ["mul0"]

def test_dead_chain_fully_removed():
    """A chain of dead ops should all be removed, not just the leaf."""
    g = make_graph(
        ("live", "MUL", ["a", "b"]),
        ("dead0", "MUL", ["a", "b"]),
        ("dead1", "ADD", ["dead0", "c"]), # depends on dead0, both must go
        outputs=["live"],
    )
    eliminate_dead_code(g)

    assert op_names(g) == ["live"]

def test_live_ops_preserved_in_original_order():
    g = make_graph(
        ("mul0", "MUL", ["a", "b"]),
        ("add0", "ADD", ["mul0", "c"]),
        ("relu0", "RELU", ["add0"]),
        outputs=["relu0"],
    )
    eliminate_dead_code(g)

    assert op_names(g) == ["mul0", "add0", "relu0"]

def test_all_ops_live_removes_nothing():
    g = make_graph(
        ("mul0", "MUL", ["a", "b"]),
        ("add0", "ADD", ["mul0", "c"]),
        outputs=["add0"],
    )
    eliminate_dead_code(g)

    assert len(g.ops) == 2

# --- inferred outputs (no explicit declaration) ---

def test_inferred_output_keeps_all_ops_when_none_are_dead():
    """Without explicit outputs, any op not consumed by another is a root."""
    g = make_graph(
        ("mul0", "MUL", ["a", "b"]),
        ("add0", "ADD", ["mul0", "c"]),
        # add0 is consumed by nobody, treated as a live output
    )
    eliminate_dead_code(g)

    assert op_names(g) == ["mul0", "add0"]

# --- edge cases ---

def test_empty_graph_returns_unchanged():
    g = make_graph(
        outputs=["out"]
    )
    eliminate_dead_code(g)

    assert g.ops == []

def test_single_live_op_survives():
    g = make_graph(
        ("out", "RELU", ["x"]),
        outputs=["out"],
    )
    eliminate_dead_code(g)

    assert op_names(g) == ["out"]

def test_output_declared_but_not_in_graph_is_ignored_gracefully():
    """If outputs references a name that doesn't exist, DCE should not crash."""
    g = make_graph(
        ("mul0", "MUL", ["a", "b"]),
        outputs=["mul0", "nonexistent"],
    )
    eliminate_dead_code(g) # should not raise

    assert "mul0" in op_names(g)