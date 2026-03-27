def constant_fold(graph):
    def is_constant(val):
        try:
            int(val)
            return True
        except (ValueError, TypeError):
            return False
         
    for op in graph.ops:
        if op.inputs and all(is_constant(x) for x in op.inputs):
            int_inputs = list(map(int, op.inputs))

            if op.op_type == "ADD":
                op.op_type = "CONST"
                op.inputs = [str(sum(int_inputs))]

            elif op.op_type == "MUL":
                result = int_inputs[0]

                for val in int_inputs[1:]:
                    result *= val
                    
                op.op_type = "CONST"
                op.inputs = [str(result)]

    return graph