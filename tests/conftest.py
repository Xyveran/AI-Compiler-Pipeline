"""
Shared fixtures for the compiler test suite.

Provides lightweight graph-building helpers so indiviudal tests can express intent
(e.g. "a graph with one ADD op") without being overrun by repeating
LLGraph/LLOp construction boilerplate.
"""

import pytest
from compiler.ir.low_level_ir import LLGraph, LLOp 

def make_graph(*ops, outputs=None):
    """Build an LLGraph from a sequence of (name, op_type, inputs) tuples."""
    g = LLGraph(outputs=outputs)

    for name, op_type, inputs in ops:
        g.add_op(LLOp(name, op_type, inputs))

    return g

def op_map(graph):
    """Return {name: op} for convenient assertions."""
    return {op.name: op for op in graph.ops}

def op_names(graph):
    """Return ordered list of op names, useful for ordering assertions."""
    return [op.name for op in graph.ops]