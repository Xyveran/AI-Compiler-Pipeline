def generate_plan(graph, schedule):
    plan = []

    for op, sched in zip(graph.ops, schedule):
        plan.append({
            "output": op.name,
            "execute": op.op_type,
            "inputs": op.inputs,
            "strategy": sched["strategy"]
        })

    return plan