# HLO (model-level)

class HLOp:
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr # symbolic expression

    def __repr__(self):
        return f"{self.name} = {self.expr}"
    
class HLGraph:
    def __init__(self, outputs=None):
        self.ops = []
        self.outputs = outputs # explicit output node names; None means infer

    def add_op(self, op):
        self.ops.append(op)
    
    def __repr__(self):
        return "\n".join(str(op) for op in self.ops)