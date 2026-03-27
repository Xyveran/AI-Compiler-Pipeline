"""
Tests for compiler.passes.fusion

Operator fusion collapses a MUL -> ADD -> RELU sequence into a single 
FUSED op. These tests cover: the happy path, partial matches that should
NOT fuse, tail ops that must not be dropped, and name remapping when two
consecutive groups both fuse.
"""

from compiler.passes.fusion import fuse_ops
from tests.conftest import make_graph, op_names, op_map


# --- happy path ---

def test_mul_add_relu_fuses_to_single_op():
    g = make_graph(
        ("mul0", "MUL", ["a", "b"]),
        ("add0", "ADD", ["mul0", "c"]),
        ("relu0", "RELU", ["add0"]),
    )
    fuse_ops(g)

    assert len(g.ops) == 1
    assert g.ops[0].op_type == "FUSED"

def test_fused_op_is_named_after_mul():
    g = make_graph(
        ("mul0", "MUL", ["a", "b"]),
        ("add0", "ADD", ["mul0", "c"]),
        ("relu0", "RELU", ["add0"]),
    )
    fuse_ops(g)

    assert g.ops[0].name == "fused_mul0"

def test_fused_op_inputs_are_mul_inputs_plus_add_bias():
    g = make_graph(
        ("mul0", "MUL", ["a", "b"]),
        ("add0", "ADD", ["mul0", "c"]),
        ("relu0", "RELU", ["add0"]),
    )
    fuse_ops(g)

    assert g.ops[0].inputs == ["a", "b", "c"]

# --- non-matching sequences pass through unchanged ---

def test_add_without_preceding_mul_is_not_fused():
    g = make_graph(
        ("add0", "ADD", ["x", "y"]),
        ("relu0", "RELU", ["add0"]),
    )
    fuse_ops(g)
    
    assert op_names(g) == ["add0", "relu0"]

def test_mul_add_without_relu_is_not_fused():
    g = make_graph(
        ("mul0", "MUL", ["a", "b"]),
        ("add0", "ADD", ["mul0", "c"]),
    )
    fuse_ops(g)
    
    assert op_names(g) == ["mul0", "add0"]

# --- tail ops are never silently dropped ---

def test_trailing_op_after_fusion_is_preserved():
    g = make_graph(
        ("mul0", "MUL", ["a", "b"]),
        ("add0", "ADD", ["mul0", "c"]),
        ("relu0", "RELU", ["add0"]),
        ("add1", "ADD", ["relu0", "d"]), # tail op, should survive
    )
    fuse_ops(g)

    names = op_names(g)    
    assert "fused_mul0" in names
    assert "add1" in names

def test_single_trailing_op_not_dropped():
    g = make_graph(
        ("mul0", "MUL", ["a", "b"]),
    )
    fuse_ops(g)
    
    assert op_names(g) == ["mul0"]

# --- name remapping across consecutive fused groups ---

def test_two_consecutive_fusions_remap_inputs_correctly():
    """
    Two back-to-back MUL+ADD+RELU groups. The second group's MUL consumes
    the RELU output of the first group, after fusion that name is remapped
    to fused_mul0, so the second fused op must reference fused_mul0.
    """
    g = make_graph(
        ("mul0", "MUL", ["a", "b"]),
        ("add0", "ADD", ["mul0", "c"]),
        ("relu0", "RELU", ["add0"]),
        ("mul1", "MUL", ["relu0", "b"]),
        ("add1", "ADD", ["mul1", "4"]),
        ("relu1", "RELU", ["add1"]),
    )
    fuse_ops(g)
    
    assert len(g.ops) == 2
    assert g.ops[1].inputs[0] == "fused_mul0"

def test_empty_graph_is_handled():
    g = make_graph(
        outputs=["out"]
    )
    fuse_ops(g)

    assert g.ops == []