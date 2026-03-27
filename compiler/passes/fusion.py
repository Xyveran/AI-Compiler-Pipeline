def fuse_ops(graph):
    fused = []
    # Maps old output name -> new fused op name, for rewriting downstream inputs
    name_remap = {}
    i = 0

    while i < len(graph.ops):
        # Need at least 3 ops remaining to attempt fusion
        if i + 2 < len(graph.ops):
            n1, n2, n3 = graph.ops[i], graph.ops[i+1], graph.ops[i+2]

            if n1.op_type == "MUL" and n2.op_type == "ADD" and n3.op_type == "RELU":
                fused_name = f"fused_{n1.name}"
                # Rewrite inputs using any previously remapped names
                inputs = [name_remap.get(x, x) for x in n1.inputs + [n2.inputs[1]]]

                fused.append(type(n1)(
                    name=fused_name,
                    op_type="FUSED",
                    inputs=inputs
                ))
                # Record that all three old names now resolve to the fused op
                name_remap[n1.name] = fused_name
                name_remap[n2.name] = fused_name
                name_remap[n3.name] = fused_name
                i += 3
                continue
            
        # No fusion - carry the op through unchanged
        op = graph.ops[i]
        op.inputs = [name_remap.get(x, x) for x in op.inputs]
        fused.append(op)
        i += 1

    graph.ops = fused
    return graph