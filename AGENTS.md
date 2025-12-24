# Evolvo AI Reference

Quick guidance for AI assistants working in this repository.

- Source of truth: `papers/GFSL-definition.md` defines the GFSL slot semantics and constraints.
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
