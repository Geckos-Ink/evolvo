This file contains the directives for AIs, and has to be updated by AIs itself to containing all strictly needed informations about the entire project to avoid repeated studies and next steps to do. In the meanwhile, update project's README.md

# Evolvo AI Reference

Quick guidance for AI assistants working in this repository.

## Source of truth
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
  - `executor.py` - GFSLExecutor (runtime execution).
  - `evaluator.py` - RealTimeEvaluator.
  - `model.py` - RecursiveModelBuilder.
  - `evolver.py` - GFSLEvolver.
  - `qlearning.py` - GFSLQLearningGuide.
  - `supervised.py` - GFSLFeatureExtractor + GFSLSupervisedGuide.
  - `demos.py` - example_formula_discovery + example_neural_architecture_search.
- `example.py` and `examples/` scripts auto-insert `src/` into `sys.path` for local runs.

## Usage notes
- When running ad-hoc scripts from the repo, set `PYTHONPATH=src` or use the provided scripts that bootstrap the path.
- `custom_operations` is a global registry; `ValueEnumerations` consults it for custom value enumerations.
- `src/evolvo/__init__.py` re-exports the public API; prefer `from evolvo import ...`.

## Optional dependencies
- `torch` (including `torch.nn`, `torch.nn.functional`, and `torch.optim`) is optional. The supervised guidance stack (`src/evolvo/supervised.py`), the neural model builder (`src/evolvo/model.py`), GPU-aware demos, and the provided example scripts now raise informative errors when PyTorch is missing so the rest of the library can be imported with only NumPy installed.

## GFSL slot semantics and validity
- Instruction layout: fixed per genome, default is 7 slots (2-slot address + op + 2-slot address + 2-slot address), auto-sized to the maximum declared expression length.
- Slot order: target_cat, target_spec, op, source1_cat, source1_spec, source2_cat, source2_spec.
- Address encoding: use `pack_type_index` and `unpack_type_index` for variable/constant slots.
- Slot sizing: `GFSLInstruction` infers slot count from the provided list; `GFSLGenome` auto-expands to the max length.
- Fixed sizing: pass `auto_slot_count=False` or `slot_count=...` on `GFSLGenome` to lock the size.
- Validity rules: always use `SlotValidator.get_valid_options` and prefer `SlotValidator.choose_option` or
  `SlotValidator.viable_options` to avoid dead-end slot choices.
- Probability support: `SlotValidator.option_success_probabilities` and `SlotValidator.build_probability_tree`
  provide success likelihoods per option.
- SET operations: source1 is CONFIG property, source2 is VALUE from the property-specific enumeration.

## Suggested verification
- No automated test suite; run `python example.py` and `python examples/*.py` for smoke coverage.
