"""
Tests for compiler.passes.constant_folding

Constant folding should evaluate ops whose inputs are all integer literals
at compile time, replacing them with a CONST op. Ops with any variable
input must be left unchanged.
"""

import pytest
from compiler.passes.constant_folding import constant_fold
from tests.conftest import make_graph, op_map


# --- ADD folding ---

def test_add_two_constants_folds():
    g = make_graph(("add0", "ADD", ["3", "4"]))
    constant_fold(g)

    op = op_map(g)["add0"]
    assert op.op_type == "CONST"
    assert op.inputs == ["7"]

def test_add_with_variable_does_not_fold():
    g = make_graph(("add0", "ADD", ["x", "4"]))
    constant_fold(g)

    assert op_map(g)["add0"].op_type == "ADD"

# --- MUL folding ---

def test_mul_two_constants_folds():
    g = make_graph(("mul0", "MUL", ["3", "5"]))
    constant_fold(g)

    op = op_map(g)["mul0"]
    assert op.op_type == "CONST"
    assert op.inputs == ["15"]

def test_mul_with_variable_does_not_fold():
    g = make_graph(("mul0", "MUL", ["a", "5"]))
    constant_fold(g)

    assert op_map(g)["mul0"].op_type == "MUL"

# --- negative numbers ---

def test_add_negative_constant_folds():
    g = make_graph(("add0", "ADD", ["-2", "5"]))
    constant_fold(g)

    op = op_map(g)["add0"]
    assert op.op_type == "CONST"
    assert op.inputs == ["3"]

def test_mul_negative_constant_folds():
    g = make_graph(("mul0", "MUL", ["-3", "4"]))
    constant_fold(g)

    op = op_map(g)["mul0"]
    assert op.op_type == "CONST"
    assert op.inputs == ["-12"]

# --- other op types are untouched ---

def test_relu_with_constant_input_not_folded():
    """Constant folding only handles ADD and MUL, RELU is left to the runtime."""
    g = make_graph(("relu0", "RELU", ["5"]))
    constant_fold(g)

    assert op_map(g)["relu0"].op_type == "RELU"

# --- empty inputs guard ---

def test_op_with_no_inputs_is_skipped():
    g = make_graph(("op0", "ADD", []))
    constant_fold(g) # should not raise

    assert op_map(g)["op0"].op_type == "ADD"

# --- graph with mixed ops, only constants fold ---

def test_only_all_constant_ops_fold():
    g = make_graph(
        ("add0", "ADD", ["2", "3"]), # folds
        ("add1", "ADD", ["x", "3"]), # does not fold
        ("mul0", "MUL", ["4", "5"]), # folds
    )
    constant_fold(g)

    ops = op_map(g)
    assert op_map(g)["add0"].op_type == "CONST"
    assert op_map(g)["add1"].op_type == "ADD"
    assert op_map(g)["mul0"].op_type == "CONST"