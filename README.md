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
```

> The core library only depends on NumPy. PyTorch is optional and only required for supervised guidance, neural architecture assembly, or demos that call `torch`.

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

---

## Library Tour

### GFSL Core

| Component | Purpose |
|-----------|---------|
| `GFSLInstruction` | Fixed-slot instruction representation (default 7 slots, auto-sized as needed). |
| `SlotValidator` | Enforces cascading slot validity and tracks active variables/constants. |
| `ValueEnumerations` | Operation- and config-specific lookup tables for numeric slots. |
| `GFSLGenome` | Holds instructions, output declarations, signatures, and operation/effective extraction helpers. |
| `GFSLExecutor` | Executes only the effective instructions while supporting injected inputs. |

### Evaluation & Search

| Component | Purpose |
|-----------|---------|
| `RealTimeEvaluator` | Runs genomes across multiple test cases, aggregates scores, and supports callbacks per evaluation. |
| `RecursiveModelBuilder` | Translates neural GFSL genomes into PyTorch `nn.Module`s (e.g., CNNs). |
| `GFSLEvolver` | Population management (init/mutate/crossover/selection), integrates optional guidance, and tracks diversity. |

### Guidance Strategies

- `GFSLQLearningGuide` — a tabular, slot-wise Q-learning agent that chooses valid slot values.
- `GFSLSupervisedGuide` — a learnable meta-model:
  - Extracts structural features from genomes (`GFSLFeatureExtractor`).
  - Learns a regression model (`GFSLSupervisedDirectionModel`) that predicts future fitness.
  - Biases mutation by sampling several candidate offspring and choosing the one with the best predicted fitness.

Both approaches respect GFSL’s fixed-slot constraints; the supervised guide simply tilts the mutation operator toward more promising instruction combinations.

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

## Practical Examples

### 1. Formula Discovery (Algorithms)

```python
from evolvo import (
    DataType, GFSLExecutor, GFSLEvolver, RealTimeEvaluator
)

test_cases = [{"d$0": float(x)} for x in range(-5, 6)]
expected = [{"d$1": float(x**2 + 2*x + 1)} for x in range(-5, 6)]

evaluator = RealTimeEvaluator(test_cases, expected)
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
```

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
