from compiler.ir.low_level_ir import LLGraph, LLOp

def lower_to_ll(hlo_graph):
    ll = LLGraph(outputs=hlo_graph.outputs)

    for op in hlo_graph.ops:
        expr = op.expr.strip()
        name = op.name # preserve the HLO name so downstream ops can reference it

        if expr.startswith("relu(") and expr.endswith(")"):
            inner = expr[len("relu("):-1].strip()
            ll.add_op(LLOp(name, "RELU", [inner]))

        elif "*" in expr:
            a, b = expr.split("*", 1)
            ll.add_op(LLOp(name, "MUL", [a.strip(), b.strip()]))

        elif "+" in expr:
            a, b = expr.split("+", 1)
            ll.add_op(LLOp(name, "ADD", [a.strip(), b.strip()]))

        else:
            raise ValueError(f"Unsupported expression in op '{op.name}': '{expr}'")
        
    return ll