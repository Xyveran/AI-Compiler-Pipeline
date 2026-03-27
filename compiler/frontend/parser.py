# Model -> IR

import json
from compiler.ir.high_level_ir import HLGraph, HLOp

def parse_model(path):
    with open(path) as f:
        data = json.load(f)

    # Support both legacy flat list and new {outputs, ops} format
    if isinstance(data, list):
        ops = data
        outputs = None
    else:
        ops = data["ops"]
        outputs = data.get("outputs")

    graph = HLGraph(outputs=outputs)
    for item in ops:
        graph.add_op(HLOp(item["name"], item["expr"]))

    return graph