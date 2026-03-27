"""
Dead Code Elimination (DCE)
===========================
Removes ops whose outputs are never consumed by any other op or graph output.

Algorithm:
    1. Determine roots, the live outputs of the graph
        - If the graph declares explicit output names, use those
        - Otherwise fall back to inferring: any op whose output is not consumed
          by another op is treated as a root. (May miss dead chains that terminate
          in an unused value, we use explicit outputs to avoid this.)
    2. Build a map of { op_name -> op } for 0(1) lookup.
    3. Walk backwards from each root (reverse data-flow), marking every op
       that a live op transitively depends on.
    4. Reconstruct the op list in original order, keeping only live ops.

This is a conservative liveness analysis, equivalent to a simplified version of the
backward liveness pass in SSA-form compilers (e.g. LLVM's DeadCodeElim).
A production implementation would also remove unused function arguments
and propagate liveness across basic block boundaries.
"""

def eliminate_dead_code(graph):
    if not graph.ops:
        return graph
    
    # Step 1: name -> op index for reverse lookup
    op_by_name = {op.name: op for op in graph.ops}

    # if graph.outputs:
    #     missing = [name for name in graph.outputs if name not in op_by_name]
    #     if missing:
    #         raise ValueError(f"Unknown outputs: {missing}")

    # Step 2: determine roots
    if graph.outputs:
        # Explicit outputs: only trust what the model declares as live
        roots = [op_by_name[name] for name in graph.outputs if name in op_by_name]
    else:
        # Inferred outputs: ops whose output is never consumed by another op
        used_as_input = {input for op in graph.ops for input in op.inputs}
        roots = [op for op in graph.ops if op.name not in used_as_input]

    # Step 3: backward reachability from roots
    live = set()

    def mark_live(op):
        if op.name in live:
            return
        live.add(op.name)
        for input in op.inputs:
            if input in op_by_name:
                mark_live(op_by_name[input])

    for root in roots:
        mark_live(root)

    # Step 4: filter, preserving original order
    removed = [op.name for op in graph.ops if op.name not in live]
    graph.ops = [op for op in graph.ops if op.name in live]

    if removed:
        print(f" [DCE] Removed {len(removed)} dead op(s): {', '.join(removed)}")
    else:
        print(" [DCE] No dead ops found.")

    return graph
