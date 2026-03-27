"""Tests for compiler.passes.pass_manager

The PassManager's contract: passes run in registration order, each recieves
the graph returned by the previous pass, and the final graph is returned.
dump_it should not affect correctness.
"""

from compiler.passes.pass_manager import PassManager
from tests.conftest import make_graph, op_names


def tag_pass(tag):
    """Returns a pass function that appends a tag to every op name."""
    def _pass(graph):
        for op in graph.ops:
            op.name = f"{op.name}_{tag}"

        return graph
    
    return _pass

# --- ordering ---

def test_passes_run_in_registration_order():
    """Each pass sees the graphs as left by the previous one."""
    g = make_graph(("op", "ADD", ["x", "y"]))

    pm = PassManager()
    pm.add_pass(tag_pass("first"), name="first")
    pm.add_pass(tag_pass("second"), name="second")

    result = pm.run(g)
    # first renames op -> op_first, then second renames that -> op_first_second
    assert op_names(result) == ["op_first_second"]

def test_single_pass_runs():
    g = make_graph(("op", "MUL", ["a","b"]))

    pm = PassManager().add_pass(tag_pass("x"))
    pm.run(g)

    assert op_names(g) == ["op_x"]

def test_no_passes_returns_graph_unchanged():
    g = make_graph(("op", "ADD", ["x","y"]))

    pm = PassManager()
    result = pm.run(g)

    assert op_names(result) == ["op"]

# --- fluent chaining ---

def test_add_pass_returns_self_forc_chaining():
    pm = PassManager()
    result = pm.add_pass(tag_pass("a"))

    assert result is pm

def test_fluent_chain_registers_all_passes():
    g = make_graph(("op", "ADD", ["x","y"]))

    pm = (
        PassManager()
            .add_pass(tag_pass("a"))
            .add_pass(tag_pass("b"))
            .add_pass(tag_pass("c"))
    )
    pm.run(g)

    assert op_names(g) == ["op_a_b_c"]

# --- dump_ir correctness ---

def test_dump_ir_does_not_affect_output(capsys):
    g = make_graph(("op", "MUL", ["a","b"]))

    pm = PassManager(
        dump_ir=True
        ).add_pass(tag_pass("x"), name="test-pass")
    
    result = pm.run(g)

    assert op_names(result) == ["op_x"]
    captured = capsys.readouterr()
    assert "test-pass" in captured.out

def test_dump_ir_prints_pass_name(capsys):
    g = make_graph(("op", "ADD", ["x","y"]))

    pm = PassManager(
        dump_ir=True
        ).add_pass(tag_pass("y"), name="my-pass")
    
    pm.run(g)

    assert "my-pass" in capsys.readouterr().out

def test_dump_ir_on_empty_graph_does_not_crash(capsys):
    g = make_graph()

    pm = PassManager(
        dump_ir=True
        ).add_pass(lambda g: g, name="noop")
    
    pm.run(g) # should not raise