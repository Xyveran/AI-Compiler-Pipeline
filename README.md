## AI Compiler Pipeline

A toy AI compiler implementing the full model → IR → optimization → codegen stack,
inspired by production compilers like MLIR and TVM.
Built as a ground-up exploration of how AI workloads are lowered and optimized for hardware execution.
JSON Model → Parser → HLGraph → Lowering → LLGraph → PassManager → Schedule → Codegen → Executor

#### Quick Start

```bash 
# Clone and run — no dependencies beyond Python 3.10+
git clone https://github.com/Xyveran/AI-Compiler-Pipeline.git
cd AI-Compiler-Pipeline

# Unoptimized run
python main.py --input examples/model.json

# With optimization passes (constant folding → DCE → operator fusion)
python main.py --input examples/model.json --optimize

# Dump IR after each pass (useful for debugging pass ordering)
python main.py --input examples/model_with_dce.json --optimize --dump-ir

# Run the test suite
pip install pytest
python -m pytest tests/ -v
```

#### Pipeline Stages

##### 1 — Frontend (Parser)

Ingests a JSON computation graph into a High-Level IR (HLGraph). Supports both
a legacy flat list format and a structured { outputs, ops } format. The outputs
field declares which op names are live graph outputs. This is required for precise
Dead Code Elimination and mirrors how ONNX and MLIR module ops
declare result values.

##### 2 — Lowering

Translates each symbolic HLO expression ("a * b", "relu(x)") into a typed
LLOp with explicit inputs. Names are preserved through lowering so downstream
ops can reference them by name, equivalent to SSA value naming in LLVM IR.
Unsupported expressions raise rather than silently drop ops.

##### 3 — Optimization Passes (PassManager)

Passes run through a PassManager that enforces a clean graph → graph contract.
Each pass is a plain function, making passes trivially testable in isolation and
easy to compose. This mirrors the design of MLIR's PassManager and LLVM's
FunctionPassManager.

Three passes are implemented:

###### Constant Folding

Evaluates ops whose inputs are all integer literals at compile time, replacing
them with a CONST op. Handles both ADD and MUL, including negative integers.
Ops with any variable input are left unchanged.

> add1: ADD(mul1, 4)   →   add1: CONST(34)   # when mul1 is also constant

###### Dead Code Elimination (DCE)

Removes ops whose outputs are never transitively consumed by a live graph output.
Uses a recursive backward liveness walk from the declared output roots, equivalent
to a simplified version of the backward liveness pass in SSA-form compilers
(e.g. LLVM's DeadCodeElim). Falls back to inferring roots from graph topology
when no explicit outputs are declared.

> dead_mul: MUL(a, b)       ← never consumed
> dead_add: ADD(dead_mul, c) ← never consumed
> Both removed. Full dead chains are eliminated, not just leaf ops.

###### Operator Fusion

Collapses MUL → ADD → RELU sequences into a single FUSED op, reducing
kernel launch overhead and enabling register-level data reuse. Rewrites downstream
input references using a name remap table so fused output names propagate
correctly through the rest of the graph. This is the same SSA value-replacement
pattern used in LLVM's InstCombine.

> mul0: MUL(a, b)       \
> add0: ADD(mul0, c)    |  →  fused_mul0: FUSED(a, b, c)
> relu0: RELU(add0)     /

The --dump-ir flag prints the IR after each pass, equivalent to LLVM's
--print-after-all:

> PassManager - IR after 'constant-folding':
>   mul0: MUL(a, b)
>   ...
>
> DCE - Removed 2 dead op(s): dead_mul, dead_add
> 
> PassManager - IR after 'dead-code-elimination':
>   mul0: MUL(a, b)
>   ...
>
> PassManager - IR after 'operator-fusion':
>   fused_mul0: FUSED(a, b, c)
>   fused_mul1: FUSED(fused_mul0, b, 4)

##### 4 — Scheduler

Assigns an execution strategy to each op. Fused ops are marked vectorized;
all others are sequential. In a production compiler this stage would model
hardware resources, tile sizes, and memory hierarchy to determine parallelism.

##### 5 — Code Generation

Produces a flat execution plan. An ordered list of steps with resolved input
names, op types, and strategies. Designed to be straightforward to lower further
to LLVM IR, CUDA kernels, or hardware-specific instruction streams.

##### 6 — Executor

A reference interpreter that evaluates the execution plan against concrete inputs.
Supports ADD, MUL, RELU, FUSED, and CONST op types. Used in tests to
verify that the optimized and unoptimized paths produce identical outputs, the
strongest correctness guarantee a compiler can provide.

#### Example Output

##### Unoptimized (python main.py --input examples/model.json):
> --- [2] Lowered IR ---
> mul0: MUL(a, b)
> add0: ADD(mul0, c)
> relu0: RELU(add0)
> mul1: MUL(relu0, b)
> add1: ADD(mul1, 4)
> relu1: RELU(add1)

> --- [6] Execution (inputs: {'a': 2, 'b': 3, 'c': 4}) ---
> mul0 = 6
> add0 = 10
> relu0 = 10
> mul1 = 30
> add1 = 34
> relu1 = 34

##### Optimized (--optimize, using model_with_dce.json which has 8 ops):
> --- [3] Optimization Passes ---
> [DCE] Removed 2 dead op(s): dead_mul, dead_add
>
>  IR after all passes:
>  fused_mul0: FUSED(a, b, c)
>  fused_mul1: FUSED(fused_mul0, b, 4)
>
> --- [6] Execution (inputs: {'a': 2, 'b': 3, 'c': 4}) ---
> fused_mul0 = 10
> fused_mul1 = 34

8 ops → 2 ops. Same output.

##### Tests

49 tests across 5 modules, covering each pass in isolation and the full
pipeline end-to-end:

> tests/
>  conftest.py               # shared graph-building fixtures
>  test_lower.py             # expression lowering, name preservation, error handling
>  test_constant_folding.py  # ADD/MUL folding, negative numbers, mixed graphs
>  test_fusion.py            # happy path, partial matches, tail ops, name remapping
>  test_dce.py               # explicit/inferred outputs, dead chains, edge cases
>  test_pass_manager.py      # ordering, fluent chaining, dump_ir correctness
>  test_pipeline.py          # end-to-end correctness, optimized == unoptimized

The most important test is test_optimized_and_unoptimized_agree(). It asserts
that the optimizer is semantics-preserving, which is the core correctness
invariant of any compiler.

```bash
python -m pytest tests/ -v
# 49 passed in 0.22s
```

#### Design Notes

Passes are plain functions, not classes. The graph → graph contract means
any function that takes an LLGraph and returns one is a valid pass. The
PassManager handles ordering and diagnostics; passes handle nothing else. This
makes each pass independently unit-testable with zero framework overhead.
Names survive lowering. lower_to_ll() preserves HLO op names into the
LLGraph rather than generating fresh t0, t1, ... temporaries. This keeps
the IR human-readable and means downstream references ("mul0 + c" → inputs
["mul0", "c"]) resolve without a separate symbol table.
Explicit outputs unlock precise DCE. Without declaring which ops are live
outputs, DCE can only infer roots by topology and a dead chain that produces
a terminal value looks identical to a live output. The outputs field in the
model format solves this cleanly, and threads through HLGraph → LLGraph so
it's available to passes after lowering.

##### What's Next

A natural extension of this pipeline toward production would include:

- ONNX ingestion: replace the JSON frontend with an ONNX parser so real
  model weights and operator types can be consumed directly
- MLIR lowering: emit MLIR dialects (linalg, affine) instead of the
  custom LLGraph, enabling the full MLIR optimization and backend ecosystem
- Memory planning pass: assign static buffer offsets to op outputs,
  eliminating dynamic allocation during inference, critical for embedded
  targets with no heap
- Tiling and vectorization: extend the scheduler to model SIMD width and
  cache tile sizes for the target NPU or DSP, rather than the binary
  sequential / vectorized strategy used here
- Cost model: score candidate fusion groups by estimated FLOPs and memory
  traffic so the fusion pass can make data-driven decisions rather than 
  matching fixed patterns