# Execution engine

def run(plan, inputs):
    results = dict(inputs)

    for step in plan:
        args = []
        for x in step["inputs"]:
            if x in results:
                args.append(results[x])
            else:
                try:
                    args.append(int(x))
                except ValueError:
                    raise KeyError(f"Input '{x}' not found in results or constants")

        if step["execute"] == "ADD":
            out = args[0] + args[1]
        elif step["execute"] == "MUL":
            out = args[0] * args[1]
        elif step["execute"] == "RELU":
            out = max(0, args[0])
        elif step["execute"] == "FUSED":
            out = max(0, (args[0] * args[1] + args[2]))
        elif step["execute"] == "CONST":
            out = args[0]
        else:
            continue
        
        results[step.get("output", f"tmp_{len(results)}")] = out
        
    return results