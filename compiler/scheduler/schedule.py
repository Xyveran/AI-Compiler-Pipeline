def create_schedule(graph):
    schedule = []

    for op in graph.ops:
        if op.op_type == "FUSED":
            strategy = "vectorized"
        else:
            strategy = "sequential"

        schedule.append({
            "op": op.name,
            "type": op.op_type,
            "strategy": strategy
        })
    
    return schedule