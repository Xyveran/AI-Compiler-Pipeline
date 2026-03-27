#LLO (lowered ops)

class LLOp:
    def __init__(self, name, op_type, inputs):
        self.name = name
        self.op_type = op_type
        self.inputs = inputs

    def __repr__(self):
        return f"{self.name}: {self.op_type}({', '.join(self.inputs)})"
    
class LLGraph:
    def __init__(self, outputs=None):
        self.ops = []
        self.outputs = outputs # explicit output node names; None means infer

    def add_op(self, op):
        self.ops.append(op)
    
    def __repr__(self):
        return "\n".join(str(op) for op in self.ops)