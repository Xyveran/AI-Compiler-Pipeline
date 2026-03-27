"""
PassManager
===========
Runs a sequence of graph optimization passes, each with the contract:

    pass_fn(graph: LLGraph) -> LLGraph
    
Passes are applied in registration order. The PassManager optionally dumps
the IR after each pass for debugging, mirroring the --print-after-all flag
in LLVM and the debug IR printing built into MLIR's pass pipeline.

Design note: keeping each pass as a plain graph -> graph function
(rather than a class) makes passes trivial to test in isolation and easy to
compose. The PassManager is the only place that knows about ordering and diagnostics.
"""

class PassManager():

    def __init__(self, dump_ir: bool = False):
        self._passes: list = []
        self.dump_ir = dump_ir

    def add_pass(self, fn, name: str = None) -> "PassManager":
        """ Register a pass. Returns self for fluent chaining."""
        self._passes.append((name or fn.__name__, fn))
        return self

    def run(self, graph):
        """Apply all registered passes in order. Returns the final graph."""
        for pass_name, fn in self._passes:
            graph = fn(graph)

            if self.dump_ir:
                print(f"\n [PassManager] IR after '{pass_name}':")

                if graph.ops:
                    for op in graph.ops:
                        print(f" {op}")

                else:
                    print(f" (empty graph)")
                    
        return graph

