# CLI entrypoint

import argparse
from compiler.frontend.parser import parse_model
from compiler.lowering.lower import lower_to_ll
from compiler.passes.fusion import fuse_ops
from compiler.passes.constant_folding import constant_fold
from compiler.passes.dce import eliminate_dead_code
from compiler.passes.pass_manager import PassManager
from compiler.scheduler.schedule import create_schedule
from compiler.codegen.codegen import generate_plan
from compiler.runtime.executor import run


def main():
    args = parse_args()

    print("\n========== AI Compiler Pipeline ==========")

    # --- Frontend: parse model into high-level IR ---
    hlo = parse_model("args.input")
    print("\n--- [1] High-Level IR (HLO) ---")
    print(hlo)

    # --- Lowering: HLO -> low-level ops ---
    ll = lower_to_ll(hlo)
    print("\n--- [2] Lowered IR ---")
    print(ll)

    # --- Optimization passes ---
    if args.optimize:
        print("\n--- [3] Optimization Passes ---")
        pm = (
            PassManager(dump_ir=args.dump_ir)
                .add_pass(constant_fold,       name="constant-folding")
                .add_pass(eliminate_dead_code, name="dead-code-elimination")
                .add_pass(fuse_ops,            name="operator-fusion")
        )
        ll = pm.run(ll)

        if not args.dump_ir:
            print("\n IR after passes:")
            for op in ll.ops:
                print(f" {op} ")
    else:
        print("\n--- [3] Optimization Passes (skipped, pass --optimize to enable) ---")

    # --- Scheduling ---
    schedule = create_schedule(ll)
    print("\n--- [4] Schedule ---")
    for entry in schedule:
        print(f" {entry['op']:20s} type={entry['type']:10s} strategy={entry['strategy']}")

    # --- Code generation
    plan = generate_plan(ll, schedule)
    print("\n--- [5] Execution Plan ---")
    for step in plan:
        print(f" {step['execute']:10s} inputs={step['inputs']} strategy={step['strategy']}")

    # --- Execution ---
    inputs = {"a": 2, "b": 3, "c": 4}
    print("\n--- [6] Execution (inputs: {inputs}) ---")
    output = run(plan, inputs)
    for k, v in output.items():
        if k not in inputs:
            print(f" {k} = {v}")

print("\n========================================\n")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mini AI compiler pipeline (MLIR-inspired)"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to model JSON file (e.g. examples/model.json)"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Run optimization passes (constant folding, DCE, op fusion)"
    )
    parser.add_argument(
        "--dump-ir", action="store_true",
        help="Print IR after each optimization pass (requires --optimize)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()