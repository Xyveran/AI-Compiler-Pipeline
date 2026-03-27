"""
Tests for compiler.lowering.lower

Verifies that each HLO expression type lowers to the correct LLOp,
that HLO names are preserved (not renamed to tN), and that unsupported
expressions raise rather than silently drop ops.
"""

import pytest
from compiler.ir.high_level_ir import HLGraph, HLOp
from compiler.lowering.lower import lower_to_ll


def make_hlo(*ops, outputs=None):
    g = HLGraph(outputs=outputs)

    for name, expr in ops:
        g.add_op(HLOp(name, expr))

    return g

# --- expression types ---

def test_mul_lowers_correctly():
    ll = lower_to_ll(make_hlo(("mul0", "a * b")))
    assert len(ll.ops) == 1

    op = ll.ops[0]
    assert op.op_type == "MUL"
    assert op.inputs == ["a", "b"]

def test_add_lowers_correctly():
    ll = lower_to_ll(make_hlo(("add0", "x + y")))

    op = ll.ops[0]
    assert op.op_type == "ADD"
    assert op.inputs == ["x", "y"]

def test_relu_lowers_correctly():
    ll = lower_to_ll(make_hlo(("relu0", "relu(add0)")))

    op = ll.ops[0]
    assert op.op_type == "RELU"
    assert op.inputs == ["add0"]

def test_relu_strips_whitespace():
    ll = lower_to_ll(make_hlo(("relu0", "relu( add0 )")))
    assert ll.ops[0].inputs == ["add0"]

def test_all_three_types_in_sequence():
    ll = lower_to_ll(make_hlo(
        ("mul0", "a * b"),
        ("add0", "mul0 + c"),
        ("relu0", "relu(add0)"),
    ))

    types = [op.op_type for op in ll.ops]
    assert types == ["MUL", "ADD", "RELU"]

# --- name preservation ---

def test_hle_names_are_preserved():
    """Lowered ops must keep their HLO names so downstream ops can reference them."""
    ll = lower_to_ll(make_hlo(
        ("mul0", "a * b"),
        ("add0", "mul0 + c"),
        ("relu0", "relu(add0)"),
    ))

    assert [op.name for op in ll.ops] == ["mul0", "add0", "relu0"]

# --- outputs field threads through ---

def test_outputs_propagated_to_ll_graph():
    hlo = make_hlo(("relu0", "relu(x)"), outputs=["relu0"])
    ll = lower_to_ll(hlo)

    assert ll.outputs == ["relu0"]

# --- error handling ---

def test_unsupported_expression_raises():
    with pytest.raises(ValueError, match="Unsupported expression"):
        lower_to_ll(make_hlo(("op0", "sin(x)")))