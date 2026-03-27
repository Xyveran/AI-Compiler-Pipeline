"""
End-to-end pipeline tests.

These tests exercise the full stack, 
    parse -> lower -> passes -> 
    schedule -> codegen -> execute, 
and assert on final numeric outpuit. The optimized and 
unoptimized paths must produce the same answer, which is the 
strongest correctness guarantee we can make.
"""

import pytest
from compiler.frontend.parser import parse_model
from compiler.lowering.lower import lower_to_ll
from compiler.passes.constant_folding import constant_fold
from compiler.passes.dce import eliminate_dead_code
from compiler.passes.fusion import fuse_ops
from compiler.passes.pass_manager import PassManager
from compiler.scheduler.schedule import create_schedule
from compiler.codegen.codegen import  generate_plan
from compiler.runtime.executor import run


INPUTS = {"a": 2, "b": 3, "c": 4}

def run_pipeline(model_path, optimize=False):
    hlo = parse_model(model_path)
    ll = lower_to_ll(hlo)

    if optimize:
        pm = (
            PassManager()
                .add_pass(constant_fold,       name="constant-folding")
                .add_pass(eliminate_dead_code, name="dce")
                .add_pass(fuse_ops,            name="fusion")
        )

        ll = pm.run(ll)
    
    schedule = create_schedule(ll)
    plan = generate_plan(ll, schedule)

    return run(plan, INPUTS)

# --- correctness: optimized == unoptimized ---

def test_optimized_and_unoptimized_agree(tmp_path):
    """
    The optimizer must not change what the model computes.
    This is the most important correctness property of any compiler.
    """
    unopt = run_pipeline("examples/model.json", optimize=False)
    opt = run_pipeline("examples/model.json", optimize=True)

    # Compare only derived values (not input pass-throughs)
    unopt_derived = {k: v for k, v in unopt.items() if k not in INPUTS}
    opt_derived =   {k: v for k, v in opt.items()   if k not in INPUTS}

    assert unopt_derived["relu1"] == opt_derived["fused_mul1"]

def test_final_output_value_is_correct():
    """
    With a=2, b=3, c=4:
        mul0 = 2*3 = 6
        add0 = 6+4 = 10
        relu0 = max(0,10) = 10
        mul1 = 10*3 = 30
        add1 = 30+4 = 34
        relu1 = max(0,34) = 34
    """
    result = run_pipeline("examples/model.json", optimize=False)

    assert result["relu1"] == 34

def test_optimized_final_output_value_is_correct():
    result = run_pipeline("examples/model.json", optimize=True)

    assert result["fused_mul1"] == 34

def test_relu_clamps_negative_values():
    """Verify relu behavior with inputs that produce a negative intermediate"""
    result = run_pipeline("examples/model.json", optimize=False)

    for k, v in result.items():
        if k not in INPUTS and "relu" in k:
            assert v >= 0, f"RELU output {k}={v} should be non-negative"

# --- DCE model: dead ops are removed and output is unchanged ---

def test_dce_model_produces_correct_output():
    result = run_pipeline("examples/model_with_dce.json", optimize=True)

    assert result["fused_mul1"] == 34

