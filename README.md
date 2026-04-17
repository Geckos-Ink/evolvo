# Evolvo

GFSL-aligned evolutionary search engine with optional supervised PyTorch guidance.

The project reimplements the **Genetic Fixed Structure Language (GFSL)** as described in the “GFSL Basic” and “GFSL Extensibility” papers. Evolvo keeps the strictly numeric, compressed instruction encoding (default 7 slots, auto-sized to the maximum declared expression length) while extending the framework with learning-based direction models that bias evolution toward higher-fitness regions.

> 📄 The original specifications live in [`papers/`](papers); start with [`GFSL-definition.md`](papers/GFSL-definition.md).

---

## Highlights

- **Fixed compressed genome language** (default 7 slots, auto-sized to max expression length) that guarantees every instruction is valid by construction.
- **Cascading slot validator** with progressive type activation (decimal → boolean → tensor) and context-aware enumerations.
- **Effective algorithm extraction** so execution only touches the instructions that matter.
- **Targeted operation extraction** to pull only the dependencies for specific result references.
- **Neural architecture support** via `RecursiveModelBuilder`, enabling GFSL to describe CNN/MLP topologies (requires PyTorch).
- **Hybrid guidance**:
  - `GFSLQLearningGuide` learns slot choices with tabular Q-learning.
  - `GFSLSupervisedGuide` is a PyTorch model that predicts fitness and proposes smarter mutations (optional; skip if PyTorch is unavailable).
- **Real-time evaluation** (`RealTimeEvaluator`) with pluggable scoring aggregators and per-test callbacks.
- **Operation weights (optional)** to attach usefulness or performance metadata to instructions or operation groups.
- **Optional typed list support** with `d!0` / `d!#0` references and dedicated queue/stack ops (`PREPEND`, `APPEND`, `CLONE`, `FIFO`, `FILO`, `LISTCOUNT`, `LISTHASITEMS`).
- **Optional typed function support** with declarations (`d&0 FUNC`), returns (`END d$k` / `END n#0`), and calls (`d$2 CALL d&0`).
- **Instruction activity tracking + pruning** so genomes can count active instructions over time and trim stale code (`prune_stale_instructions`).
- **Rich utilities** for mutation, crossover, diversity tracking, and population management.
- **Ad-hoc personalization** via `register_custom_operation`, letting you graft bespoke operations into the validator/executor without forking the core.

---

## Installation

```bash
# clone the repo
git clone https://github.com/Geckos-Ink/evolvo.git
cd evolvo

# create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# install the base runtime dependency
pip install numpy

# install PyTorch if you plan to use the supervised guide, RecursiveModelBuilder, or the torch-backed examples
pip install torch

# install optional accelerator extras (torch + kp/kompute bindings)
pip install -r requirements-accelerators.txt
```

> The core library only depends on NumPy. PyTorch and Kompute (`kp`) are optional accelerators.
> If `pip install kp` fails on your platform/toolchain, install Kompute Python bindings from source:
> `pip install git+https://github.com/KomputeProject/kompute.git`

If you are running directly from the repo (without installing a package), add the source folder to your Python path:

```bash
export PYTHONPATH=src
```

---

## Quick Start

The simplest entry point is the quadratic-formula example that couples the evolver with the supervised PyTorch guide.

```bash
python example.py
```

> This example requires PyTorch; install it only if you plan to use the supervised guide.

Typical output (truncated):

```
Gen 000 | fitness=0.214583 | guide≈      nan
Gen 010 | fitness=0.781942 | guide≈0.795312
...
Best genome fitness: 0.991332
Signature: 6c71b2e9407834ab...

Effective instruction trace:
  ✓ d#0 = 1.0
  ✓ d$2 = MUL(d$0, d$0)
  ✓ d$3 = MUL(d#1, d$0)
  ✓ d$1 = ADD(d$2, d$3)
  ✓ d$1 = ADD(d$1, d#0)
```

The demo:

1. Seeds reproducible randomness.
2. Builds a dataset for `y = 0.5x² − x + 2`.
3. Trains the `GFSLSupervisedGuide` on-the-fly as new genomes are evaluated.
4. Evolves 60 generations with real-time progress prints.
5. Displays the effective GFSL instruction trace and spot-check predictions.

> Want to see runtime personalization instead? Run `python example.py personalization` to register and execute a bespoke decimal operation.
>
> Want to see functions + pruning flow? Run `python example.py function`.
>
> Nested functions smoke demo: `python example.py nested`
>
> Void function external-write smoke demo: `python example.py void`

If you prefer to start from scratch, the minimal API looks like:

```python
from evolvo import (
    DataType, GFSLEvolver, RealTimeEvaluator, GFSLSupervisedGuide
)

evolver = GFSLEvolver(population_size=30,
                      supervised_guide=GFSLSupervisedGuide())
evolver.initialize_population("algorithm", initial_instructions=12)

for genome in evolver.population:
    genome.mark_output(DataType.DECIMAL, 1)

def fitness_fn(genome):
    return evaluator.evaluate(genome)

evolver.evolve(80, fitness_fn)
best = evolver.population[0]
print(best.to_human_readable())
```

---

## Source Layout

- `src/evolvo/` - modular package code (enums, slot utilities, validator, genome, executor, evolver, guidance models).
- `example.py` and `examples/` - runnable scripts that bootstrap `src/` into `sys.path` for local runs.

---

## Additional Examples

Practical scripts live in `examples/`:

- `examples/expression_flow.py` — slot-wise builder flow, option previews, and conditional consequents.
- `examples/neural_flow.py` — SET/CONV neural flow with a consequent RELU.
- `examples/operation_extraction.py` — dependency-based extraction for specific result references.
- `examples/function_flow.py` — typed function declaration/call plus activity-based stale-instruction pruning.
- `examples/nested_function_flow.py` — nested function behavior (`allow_nested_functions=True/False`) side-by-side.
- `examples/void_external_write_flow.py` — void function behavior with isolated scope vs. external writes.

---

## Library Tour

### GFSL Core

| Component | Purpose |
|-----------|---------|
| `GFSLInstruction` | Fixed-slot instruction representation (default 7 slots, auto-sized as needed). |
| `SlotValidator` | Enforces cascading slot validity and tracks active variables/constants/lists. |
| `ValueEnumerations` | Operation- and config-specific lookup tables for numeric slots. |
| `GFSLGenome` | Holds instructions, output declarations, signatures, and operation/effective extraction helpers. |
| `GFSLExecutor` | Executes only the effective instructions, supports optional function scopes/calls, records instruction activity, and can attempt experimental Kompute backend with CPU fallback. |

### Evaluation & Search

| Component | Purpose |
|-----------|---------|
| `RealTimeEvaluator` | Runs genomes across multiple test cases, aggregates scores, and supports callbacks per evaluation. |
| `RecursiveModelBuilder` | Translates neural GFSL genomes into PyTorch `nn.Module`s (e.g., CNNs). |
| `GFSLEvolver` | Population management with adaptive mutation pressure, diversity deadlock fallback (immigrant injection), optional batch evaluation, and guidance integration. |

### Guidance Strategies

- `GFSLQLearningGuide` — a tabular, slot-wise Q-learning agent that chooses valid slot values.
- `GFSLSupervisedGuide` — a learnable meta-model:
  - Extracts structural features from genomes (`GFSLFeatureExtractor`).
  - Learns a regression model (`GFSLSupervisedDirectionModel`) that predicts future fitness.
  - Biases mutation by sampling several candidate offspring and choosing the one with the best predicted fitness.
  - Supports accelerator auto-selection via `resolve_torch_accelerator(...)` with `auto|cpu|cuda|rocm|mps`.
  - Exposes `runtime_summary()` so you can verify the requested device, resolved backend, probe tensor device, and train/predict call counters.

Both approaches respect GFSL’s fixed-slot constraints; the supervised guide simply tilts the mutation operator toward more promising instruction combinations.

### Experimental Kompute Execution

`GFSLExecutor` supports `compute_backend="auto|cpu|kompute|kompute-sim"`:

- `auto` (default): try Kompute only if runtime is available; otherwise use CPU.
- `kompute`: force Kompute attempt; on runtime/kernel errors it warns and falls back to CPU (unless `kompute_fail_hard=True`).
- `kompute-sim`: force Kompute planning/compatibility checks, then execute with simulated runtime (CPU-backed semantics).
- Native mode dispatches supported scalar stages (`DECIMAL_OPS`, boolean compare/logic) via Vulkan shaders and transparently executes unsupported stages on CPU with synchronized state.
- Before execution, a compatibility pre-check reports unsupported op names/counts (for example `CALLx1, FUNCx1`) and enables hybrid fallback when needed.
- In `auto` mode, Kompute is process-disabled after the first non-recoverable runtime failure to avoid warning spam in large evaluation loops.
- `cpu`: always CPU path.

Example:

```python
from evolvo import GFSLExecutor

executor = GFSLExecutor(
    compute_backend="kompute",
    kompute_warn_on_fallback=True,
    kompute_fail_hard=False,
)
outputs = executor.execute(genome, {"d$0": 3.0})
```

The Kompute path is planner-first and still experimental. Unsupported kernels or runtime failures emit a warning and continue on CPU fallback.
Use `kompute-sim` for fast compatibility/planning validation without native Vulkan dispatch.

---

## Operation Extraction

Use the extraction helpers to get only the minimal operations needed for one or more result references.
You can keep the original execution order, or choose a fixed slot ordering to obtain a stable,
enumerator-ordered signature for the same operation set.

```python
from evolvo import DataType, GFSLGenome

genome = GFSLGenome("algorithm")
# ... build instructions ...

# Extract by explicit references (string, tuple, or dict forms).
indices = genome.extract_operation_indices(["d$3", (DataType.DECIMAL, 4)], order="fixed")
instructions = genome.extract_operations(["d$3"], order="execution")

# If no result refs are provided, outputs are used by default.
genome.mark_output(DataType.DECIMAL, 3)
indices = genome.extract_operation_indices(order="fixed")
```

Result reference forms:
- `"d$3"` or `"b#0"`
- `"d!0"` or `"d!#0"` (typed list and typed constant-list references)
- `(DataType.DECIMAL, 3)` or `(Category.VARIABLE, DataType.DECIMAL, 3)`
- `{"category": Category.VARIABLE, "dtype": DataType.DECIMAL, "index": 3}`

Ordering options:
- `"execution"`: original instruction index order.
- `"fixed"`: stable sort by slot values (enumerator order).
- `"topological"`: dependency order with a fixed tie-break by slot values.

---

## Operation Weights (Optional)

Attach weights to individual instructions or to groups of operations to capture
usefulness heuristics, device-specific performance costs, or guidance priors.
Weights are metadata only; execution and signatures ignore them.

```python
from evolvo import GFSLGenome, Operation

genome = GFSLGenome("algorithm")
# ... add instructions ...

genome.set_instruction_weight(0, 0.75)  # per-instruction override
genome.set_instruction_weights([1, 2, 3], 0.5)

genome.operation_weights.set_operation_weight(Operation.MUL, 0.4)
genome.operation_weights.set_group(
    "heavy_ops",
    [Operation.CONV, Operation.LINEAR],
    weight=2.0,
)

print(genome.instruction_weight(0))
print(genome.to_human_readable(include_weights=True))
```

If an operation belongs to multiple groups, `OperationWeights.resolve_weight`
combines group weights using `group_reduce` (default: `"mean"`).

---

## Optional Typed Functions

Functions are represented as typed references (`<type>&<index>`) and are fully optional.

- Declare: `d&0 FUNC`
- End/return:
  - Non-void: `END d$1`
  - Void: `END n#0` (or `END NONE`)
- Call:
  - Return value: `d$2 CALL d&0`
  - Void call: `NONE CALL n&0`

Execution behavior:
- Function bodies are isolated by default (writes stay local).
- Enable external writes with `GFSLExecutor(allow_function_external_writes=True)`.
- Nested function declarations can run when `allow_nested_functions=True` (default).
- Optional guard for void functions: `require_void_external_writes=True`.

```python
from evolvo import (
    Category, DataType, GFSLExecutor, GFSLGenome, GFSLInstruction,
    Operation, pack_type_index
)

genome = GFSLGenome("algorithm")
genome.validator.variable_counts[int(DataType.DECIMAL)] = 1  # expose d$0 input

# d&0 FUNC
genome.add_instruction(GFSLInstruction([
    Category.FUNCTION, pack_type_index(DataType.DECIMAL, 0), Operation.FUNC,
    Category.NONE, 0,
    Category.NONE, 0,
]))
# d$1 = ADD(d$0, 2.0)
genome.add_instruction(GFSLInstruction([
    Category.VARIABLE, pack_type_index(DataType.DECIMAL, 1), Operation.ADD,
    Category.VARIABLE, pack_type_index(DataType.DECIMAL, 0),
    Category.VALUE, 3,
]))
# END d$1
genome.add_instruction(GFSLInstruction([
    Category.NONE, 0, Operation.END,
    Category.VARIABLE, pack_type_index(DataType.DECIMAL, 1),
    Category.NONE, 0,
]))
# d$2 = CALL d&0
genome.add_instruction(GFSLInstruction([
    Category.VARIABLE, pack_type_index(DataType.DECIMAL, 2), Operation.CALL,
    Category.FUNCTION, pack_type_index(DataType.DECIMAL, 0),
    Category.NONE, 0,
]))

genome.mark_output(DataType.DECIMAL, 2)
print(GFSLExecutor().execute(genome, {"d$0": 3.0}))  # {'d$2': 5.0}
```

---

## Activity Tracking And Pruning

`GFSLExecutor.execute(...)` records instruction activity into `genome.instruction_activity` by default.
Use this to count active code and prune stale instructions across repeated runs/evolutions.

```python
# Run genome several times...
executor = GFSLExecutor()
for x in [1.0, 2.0, 3.0]:
    executor.execute(genome, {"d$0": x})

print([m.hits for m in genome.instruction_activity])
print(genome.active_instruction_count(min_hits=1))

# Remove instructions never used recently (while preserving effective ones by default)
removed = genome.prune_stale_instructions(min_hits=1, keep_effective=True)
print("pruned:", removed)
```

---

## Optional Typed Lists

List pointers use the `!` symbol and keep the same typed-prefix style as variables/constants:
- `d!0` → mutable decimal list index 0
- `b!1` → mutable boolean list index 1
- `d!#0` → constant decimal list index 0 (clone-only source)

Supported built-ins:
- `PREPEND` / `APPEND`: write an item into a target list (target may allocate `n+1`).
- `CLONE`: copy `d!k` or `d!#k` into a new mutable list.
- `FIFO` / `FILO`: pop first/last item from a mutable list.
- `LISTCOUNT`: decimal count of items.
- `LISTHASITEMS`: boolean non-empty check.

When `FIFO`/`FILO` are executed on an empty list, the result is `VOID` (exported as `None` in outputs). Later instructions that consume that `VOID` value are skipped.

```python
from evolvo import (
    Category, DataType, GFSLExecutor, GFSLGenome, GFSLInstruction,
    Operation, pack_type_index
)

genome = GFSLGenome("algorithm")
genome.validator.activate_type(DataType.BOOLEAN)

# d!0 = APPEND(1.0)
genome.add_instruction(GFSLInstruction([
    Category.LIST, pack_type_index(DataType.DECIMAL, 0), Operation.APPEND,
    Category.VALUE, 1,  # 1.0 from default math enumeration
    Category.NONE, 0,
]))

# d$0 = FILO(d!0)
genome.add_instruction(GFSLInstruction([
    Category.VARIABLE, pack_type_index(DataType.DECIMAL, 0), Operation.FILO,
    Category.LIST, pack_type_index(DataType.DECIMAL, 0),
    Category.NONE, 0,
]))

genome.mark_output(DataType.DECIMAL, 0)
print(GFSLExecutor().execute(genome))  # {'d$0': 1.0}
```

Use `genome.seed_list_count(dtype, count, constant=True)` (or `genome.validator.seed_list_count(...)`) to expose pre-existing constant list sources (`!#`) to the validator.

---

## Practical Examples

### 1. Formula Discovery (Algorithms)

```python
from evolvo import (
    DataType, GFSLExecutor, GFSLEvolver, RealTimeEvaluator
)

test_cases = [{"d$0": float(x)} for x in range(-5, 6)]
expected = [{"d$1": float(x**2 + 2*x + 1)} for x in range(-5, 6)]

evaluator = RealTimeEvaluator(
    test_cases,
    expected,
    executor_kwargs={"compute_backend": "auto"},
)
evolver = GFSLEvolver(population_size=30)
evolver.initialize_population("algorithm", initial_instructions=15)

for genome in evolver.population:
    genome.mark_output(DataType.DECIMAL, 1)

evolver.evolve(100, evaluator.evaluate)
best = evolver.population[0]

print("Effective instructions:")
for line in best.to_human_readable():
    if line.startswith("✓"):
        print(" ", line)

executor = GFSLExecutor()
print(executor.execute(best, {"d$0": 3.0}))
```

### 2. Neural Architecture Assembly

> Running this snippet requires PyTorch because it instantiates `RecursiveModelBuilder`.

```python
import torch
from evolvo import (
    Category, ConfigProperty, DataType, GFSLGenome,
    GFSLInstruction, Operation, RecursiveModelBuilder, pack_type_index
)

genome = GFSLGenome("neural")
genome.validator.activate_type(DataType.TENSOR)

# SET CHANNELS = 64
genome.add_instruction(GFSLInstruction([
    Category.NONE, 0, Operation.SET,
    Category.CONFIG, ConfigProperty.CHANNELS,
    Category.VALUE, 5,
]))

# t$0 = CONV(t$0) followed by RELU
genome.add_instruction(GFSLInstruction([
    Category.VARIABLE, pack_type_index(DataType.TENSOR, 0), Operation.CONV,
    Category.VARIABLE, pack_type_index(DataType.TENSOR, 0),
    Category.NONE, 0,
]))
genome.add_instruction(GFSLInstruction([
    Category.VARIABLE, pack_type_index(DataType.TENSOR, 0), Operation.RELU,
    Category.VARIABLE, pack_type_index(DataType.TENSOR, 0),
    Category.NONE, 0,
]))

builder = RecursiveModelBuilder()
model = builder.build_from_genome(genome, (3, 32, 32))

x = torch.randn(1, 3, 32, 32)
print(model(x).shape)
```

### 3. Ad-hoc Operations (Personalization)

Use `register_custom_operation` when you need a one-off opcode without editing the core library. The helper wires your function into the slot validator, executor, and human-readable traces.

```python
from evolvo import (
    Category, DataType, GFSLExecutor,
    GFSLGenome, GFSLInstruction, pack_type_index, register_custom_operation,
)

opcode = register_custom_operation(
    "affine_bias",
    target_type=DataType.DECIMAL,
    function=lambda value, bias, context=None: float(value) * 1.5 + float(bias or 0.0),
    arity=2,
    value_enumeration=[-1.0, 0.0, 0.5, 2.0],  # inline VALUE slots pull from this list
    doc="Scales the input and adds a selectable bias term.",
)

genome = GFSLGenome("algorithm")
genome.add_instruction(GFSLInstruction([
    Category.VARIABLE, pack_type_index(DataType.DECIMAL, 1), opcode,
    Category.VARIABLE, pack_type_index(DataType.DECIMAL, 0),
    Category.VALUE, 3,  # chooses bias=2.0
]))
genome.mark_output(DataType.DECIMAL, 1)

executor = GFSLExecutor()
print(executor.execute(genome, {"d$0": 2.0}))  # {'d$1': 5.0}
```

The optional `context` keyword argument receives `{"executor": ..., "instruction": ...}` so custom functions can peek at runtime state when needed.

### 4. Supervised Guided Evolution (From `example.py`)

```python
from evolvo import (
    DataType, GFSLExecutor, GFSLEvolver,
    GFSLSupervisedGuide, RealTimeEvaluator
)

guide = GFSLSupervisedGuide(hidden_layers=[256, 128],
                            candidate_pool=4, min_buffer=64)
evolver = GFSLEvolver(population_size=32, supervised_guide=guide)
evolver.initialize_population("algorithm", initial_instructions=12)

for genome in evolver.population:
    genome.mark_output(DataType.DECIMAL, 1)

evolver.evolve(60, evaluator.evaluate)
best = evolver.population[0]
print(best.fitness, best.to_human_readable())
print(guide.runtime_summary())  # verify resolved backend/device and probe status
```

If you want to train the guide once per round and reuse it without per-generation retraining:

```python
guide = GFSLSupervisedGuide(device="rocm")
evolver = GFSLEvolver(
    population_size=32,
    supervised_guide=guide,
    guide_observe_each_generation=False,
)
evolver.initialize_population("algorithm", initial_instructions=12)
guide.observe_population(evolver.population)  # one warmup/train pass
evolver.evolve(60, evaluator.evaluate)
```

### 5. Kompute Planning (Experimental)

Use the Kompute planner to compose GFSL instructions into typed kernel stages with persistent/transient buffer plans.

```python
from evolvo import (
    DataType,
    GFSLKomputePlanner,
    KomputeInstructionRegistry,
    KomputeTypeSpec,
)

registry = KomputeInstructionRegistry()
registry.set_default_type(DataType.DECIMAL, KomputeTypeSpec("f32", components=1))
registry.register_binding("PCPL_HASHMIX", shader_key="pcpl.hashmix")
registry.set_operation_type_override(
    "PCPL_HASHMIX",
    target=KomputeTypeSpec("f32", components=1),
)

planner = GFSLKomputePlanner(registry=registry)
plan = planner.compose(genome, order="effective", keep_vram_state=True)
print(plan.to_dict())
```

The planner is designed for external integrations (such as `pcpl_evolvo`) where custom instructions need explicit kernel mappings and type overrides.

---

## Customisation & Extensibility

- **Operations / Data types** — extend `Operation` and `DataType` enums; the slot validator automatically respects new options.
- **Enumerations** — add or override entries in `ValueEnumerations` to open new discrete value sets (learning rates, optimizer choices, etc.).
- **Guidance** — swap in your own mutation proposal system by implementing a `propose_mutation` method similar to `GFSLSupervisedGuide`.
- **Scoring** — pass a custom `score_aggregator` into `RealTimeEvaluator` to change how per-case fitness values are combined.

Because every slot is enumerated and validated, extending the system never introduces invalid states—new concepts simply become additional discrete options inside the GFSL language.

---

## Running the Legacy Examples

To explore the classic demonstrations (without the supervised guide), run:

```bash
PYTHONPATH=src python -c "from evolvo import example_formula_discovery; example_formula_discovery()"
PYTHONPATH=src python -c "from evolvo import example_neural_architecture_search; example_neural_architecture_search()"
```

They remain useful for validating the baseline evolver and seeing the raw GFSL output traces.

---

## Roadmap Ideas

- Gradient-informed mutation proposals (hybrid neuro-evolution).
- Additional tensor ops (attention, modern normalization layers).
- Benchmarks comparing supervised guidance vs. unguided baselines.
- Operation conversion tables and device-targeted weight profiles.
- Exporters for ONNX / TorchScript from GFSL neural genomes.

---

## References

- **GFSL Basic** — foundational description of the fixed-slot language.
- **GFSL Extensibility** — discusses type activation, enumerations, and advanced operations.
- ️`papers/` — contains PDF/Markdown reproductions of the above papers along with related notes.

---

Have fun evolving! Contributions and experiment reports are welcome—open an issue or PR if you push the framework in new directions.***
