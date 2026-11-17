This file contains the directives for AIs, and has to be updated by AIs itself to containing all strictly needed informations about the entire project to avoid repeated studies and next steps to do. In the meanwhile, update project's README.md

# Evolvo AI Reference

Quick guidance for AI assistants working in this repository.

## Source of truth
- `papers/GFSL-NN.md` defines the neural-architecture language in `src/evolvo/arch/`.
- `papers/GFSL-definition.md` defines the GFSL slot semantics and constraints.

## Repo layout (src-based)
- `src/evolvo/` is the modular package.
  - `enums.py` - Category, DataType, Operation, ConfigProperty, plus op groupings.
  - `slots.py` - slot constants, names, pack/unpack helpers.
  - `custom_ops.py` - CustomOperationManager, registry, registration helpers.
  - `values.py` - ValueEnumerations (context-aware VALUE enumerations).
  - `instruction.py` - GFSLInstruction, SlotOption, describe_slot_option.
  - `validator.py` - SlotValidator (cascading validity + probability helpers).
  - `builder.py` - GFSLExpressionBuilder (slot-wise construction).
  - `genome.py` - GFSLGenome, extraction helpers, human-readable decoding.
  - `weights.py` - OperationWeights (optional per-op and group metadata).
  - `executor.py` - GFSLExecutor (runtime execution).
  - `evaluator.py` - RealTimeEvaluator.
  - `model.py` - RecursiveModelBuilder.
  - `evolver.py` - GFSLEvolver.
  - `qlearning.py` - GFSLQLearningGuide.
  - `supervised.py` - GFSLFeatureExtractor + GFSLSupervisedGuide.
  - `demos.py` - example_formula_discovery + example_neural_architecture_search.
- `src/evolvo/arch/` - GFSL-NN (pure Python except `build.py`/`train.py`, which import torch):
  - `language.py` - Shape (vec/set + origin), OpSpec/OpRegistry (primitives + macros), ArgSpec domains
    (width ladder, depth, choice, divisor, space lists), ArchSpace (cascading options, budget masks,
    repair, validate, lower, signature, to_slots, decisions), Emitter (macro expansion), Chooser.
  - `stats.py` - ContextualStats: sliding-window records, greedy adjusted-gain feature order (pinned
    features first), prefix-group aggregates, shrinkage estimate (`count` 0 for an unseen group,
    `matched`/`evidence` for the deepest observed one), JSON round trip.
  - `policy.py` - AdaptivePolicy (softmax over shrunk reward + optimism, uniform floor, prior hooks),
    size_prior.
  - `efficiency.py` - EfficiencyModel (frontier per half-octave, relative-error power-law fit,
    efficient_size, pressure).
  - `stopping.py` - PlateauStopper (plateau / hopeless at equal effort / diverged / hard cap; learned
    patience from `gap` stats; records outcomes).
  - `search.py` - ArchitectureSearch (quantile rewards, learned mutation choice, repair after every
    edit, signature cache, window adaptation, converged/budget stops), Knowledge (save/load).
  - `build.py` - GenomeModule (traceable), build_module, parameter_count, trace_module.
  - `train.py` - train_module (AdamW, best-epoch restore), TorchTask (shares the search's epoch stats).
- `example.py` and `examples/` scripts auto-insert `src/` into `sys.path` for local runs.
- `src/evolvo/__init__.py` resolves public names lazily (PEP 562 `__getattr__`); keep new exports in
  its `_LAZY` table so pure-Python subpackages import without NumPy.
- Function-focused smoke demos:
  - `examples/function_flow.py` - typed function call + activity-based pruning.
  - `examples/nested_function_flow.py` - nested function execution (enabled/disabled comparison).
  - `examples/void_external_write_flow.py` - void function external-write policy behavior.

## Usage notes
- When running ad-hoc scripts from the repo, set `PYTHONPATH=src` or use the provided scripts that bootstrap the path.
- `custom_operations` is a global registry; `ValueEnumerations` consults it for custom value enumerations.
- `src/evolvo/__init__.py` re-exports the public API lazily; prefer `from evolvo import ...`.
- `GFSLExecutor` supports `compute_backend=auto|cpu|kompute|kompute-sim`. `auto` is safest; `kompute` attempts native runtime and falls back to CPU on runtime/kernel failures (unless `kompute_fail_hard=True`); `kompute-sim` runs compatibility/planning plus CPU-backed simulated execution.
- Native Kompute runtime now supports both kp API styles: explicit sync ops (`OpSyncDevice`/`OpSyncLocal` or `OpTensorSyncDevice`/`OpTensorSyncLocal`) and shared-memory fallback when those ops are absent.
- With Kompute enabled, process-pool evaluation is auto-switched to thread mode by default (`EVOLVO_KOMPUTE_ALLOW_PROCESS_POOL=1` can re-enable process pools).
- On multi-GPU Vulkan hosts, native Kompute device selection can be overridden with env vars: `EVOLVO_KOMPUTE_DEVICE_INDEX` and `EVOLVO_KOMPUTE_QUEUE_FAMILY`.
- Typed list pointers are supported:
  - Mutable list: `d!0`, `b!1`, `t!0` (`Category.LIST`)
  - Constant list: `d!#0` (`Category.LIST_CONSTANT`, clone-only source)
- Typed function pointers are supported:
  - Function reference: `d&0`, `b&1`, `t&0`, `n&0` (`Category.FUNCTION`)
  - `n&*` represents a void function (`DataType.NONE` return).
- Operation weights are optional metadata: `GFSLInstruction.weight` or `GFSLGenome.set_instruction_weights(...)` for
  per-instruction groups, and `GFSLGenome.operation_weights` (OperationWeights) for per-op/group weighting; weights do not
  affect execution/signatures.
- Instruction activity tracking is built into the runtime:
  - `GFSLExecutor.execute(...)` records executed instruction hits on `GFSLGenome.instruction_activity`.
  - Use `GFSLGenome.record_instruction_activity(...)`, `active_instruction_count(...)`,
    `stale_instruction_indices(...)`, and `prune_stale_instructions(...)` for usage-aware cleanup.

## Optional dependencies
- `torch` (including `torch.nn`, `torch.nn.functional`, and `torch.optim`) is optional. The supervised guidance stack (`src/evolvo/supervised.py`), the neural model builder (`src/evolvo/model.py`), GPU-aware demos, and the provided example scripts now raise informative errors when PyTorch is missing so the rest of the library can be imported with only NumPy installed.
- `kompute`/`kp` is optional. `src/evolvo/kompute.py` provides operation-to-kernel composition, compatibility reports, simulated execution, and native Vulkan dispatch for supported scalar op families with synchronized CPU fallback for unsupported stages. If PyPI install fails, prefer source install: `pip install git+https://github.com/KomputeProject/kompute.git`.

## GFSL slot semantics and validity
- Instruction layout: fixed per genome, default is 7 slots (2-slot address + op + 2-slot address + 2-slot address), auto-sized to the maximum declared expression length.
- Slot order: target_cat, target_spec, op, source1_cat, source1_spec, source2_cat, source2_spec.
- Address encoding: use `pack_type_index` and `unpack_type_index` for variable/constant/list slots.
- Function references use the same packed address format and the `&` symbol (`Category.FUNCTION`).
- Slot sizing: `GFSLInstruction` infers slot count from the provided list; `GFSLGenome` auto-expands to the max length.
- Fixed sizing: pass `auto_slot_count=False` or `slot_count=...` on `GFSLGenome` to lock the size.
- Validity rules: always use `SlotValidator.get_valid_options` and prefer `SlotValidator.choose_option` or
  `SlotValidator.viable_options` to avoid dead-end slot choices.
- Probability support: `SlotValidator.option_success_probabilities` and `SlotValidator.build_probability_tree`
  provide success likelihoods per option.
- SET operations: source1 is CONFIG property, source2 is VALUE from the property-specific enumeration.
- Function operations:
  - Declaration: `<type>&<index> FUNC` opens a function scope.
  - Return/close: `END <type>$<index>` for non-void returns, `END n#0` (or `END NONE`) for void.
  - Call: `<type>$<dst> CALL <type>&<index>` assigns function output; `NONE CALL n&<index>` is valid for void functions.
  - Nested declarations are supported by the executor (toggle via `GFSLExecutor(allow_nested_functions=...)`).
- List operations:
  - `PREPEND` / `APPEND` target a typed list and insert source1.
  - `CLONE` copies from mutable list (`!`) or constant list (`!#`) into a mutable target list.
  - `FIFO` / `FILO` pop from mutable list sources.
  - `LISTCOUNT` returns decimal length, `LISTHASITEMS` returns boolean non-empty status.
  - `FIFO` / `FILO` on empty list emit runtime `VOID`; any downstream instruction reading a `VOID` input is skipped.
- Validator list gating:
  - `CLONE` appears only when at least one compatible `!` or `!#` source exists.
  - `FIFO`/`FILO` appear only when at least one compatible mutable list exists.
  - `LISTCOUNT`/`LISTHASITEMS` appear only when at least one mutable list exists.
  - `seed_list_count(dtype, count, constant=...)` can expose pre-existing runtime lists to option generation.
- Activity-aware pruning:
  - `prune_stale_instructions(min_hits=..., max_idle_ticks=..., keep_effective=True)` removes stale instructions while preserving currently effective ones by default.
  - `rebuild_validator_state()` is available to recompute validator counters after bulk edits.

## GFSL-NN invariants
- Every genome reaching evaluation went through `ArchSpace.repair` (sampling, mutation, crossover);
  `validate()` must return `[]` for all of them. Never add an op whose options are not enumerable.
- Arg option lists must be ascending in cost (the budget mask assumes the first option is cheapest).
- Rewards are attributed through `ArchSpace.decisions(genome)`, never through the sampling trace.
- Runs stopped as `hopeless` are censored: they rank but neither teach epoch stats nor enter the
  efficiency frontier. A learned value never stops a still-improving run (only the hard cap does).
- Determinism: one `random.Random(seed)` per search; candidate *i* uses torch seed `seed*1_000_003+i`.

## Suggested verification
- `PYTHONPATH=src python -m unittest discover -s tests` (GFSL-NN; torch cases self-skip).
- Legacy GFSL has no automated suite; run:
  - `python3 example.py function`
  - `python3 example.py nested`
  - `python3 example.py void`
  - `python3 examples/*.py`

## Roadmap / next steps
- GFSL-NN: masked/padded set inputs; macro mining (promote recurring elite fragments to macros);
  FLOP/latency cost terms; multi-fidelity schedules beyond hopeless stopping; ONNX export.
- Operation conversion table (map ops to alternatives per device profile using weights).
- Add recursion-guard stress tests (`max_call_depth`) and activity-pruning policy sweeps.
- Expand native Kompute coverage beyond scalar decimal/boolean kernels (lists/tensors/control-flow-aware lowering and lower-overhead persistent dispatch across repeated evaluations).
- Add per-operation Kompute bindings for custom ops registered at runtime (including context-aware validation when custom op signatures/types change).
