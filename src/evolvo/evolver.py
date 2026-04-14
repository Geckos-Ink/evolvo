"""Evolution engine for GFSL genomes."""

from __future__ import annotations

import copy
import hashlib
import random
from typing import Callable, List, Optional, Sequence, Set, TYPE_CHECKING

from .genome import GFSLGenome
from .instruction import GFSLInstruction

if TYPE_CHECKING:
    from .supervised import GFSLSupervisedGuide


_MUTATION_IMPROVEMENT_DECAY_STEP_MULTIPLIER = 0.30
_MUTATION_STAGNATION_BOOST_STEP_MULTIPLIER = 1.30
_MUTATION_SIGNATURE_STALL_BOOST_STEP_MULTIPLIER = 0.45
_MUTATION_FORCE_THRESHOLD = 0.45
_MUTATION_FORCE_CHANCE = 0.82


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _invalidate_genome_caches(genome: GFSLGenome) -> None:
    genome._signature = None
    genome._effective_instructions = None
    if hasattr(genome, "_evolvo_eval_sig"):
        delattr(genome, "_evolvo_eval_sig")
    if hasattr(genome, "_evolvo_eval_sig_key"):
        delattr(genome, "_evolvo_eval_sig_key")


def _append_random_instruction_fast(
    genome: GFSLGenome,
    *,
    max_attempts: int = 8,
) -> bool:
    """Add one random valid instruction without building probability trees."""
    slot_count = max(1, int(genome.validator.slot_count))
    for _ in range(max(1, int(max_attempts))):
        instruction = GFSLInstruction(slot_count=slot_count)
        valid = True
        for slot_idx in range(slot_count):
            options = genome.validator.get_valid_options(instruction, slot_idx)
            if not options:
                valid = False
                break
            instruction.slots[slot_idx] = random.choice(options)
        if not valid:
            continue

        genome.instructions.append(instruction)
        genome.validator.update_state(instruction)
        try:
            genome._ensure_activity_size()  # type: ignore[attr-defined]
        except Exception:
            pass
        _invalidate_genome_caches(genome)
        return True
    return False


def _evaluation_signature(genome: GFSLGenome) -> str:
    """Fast signature used for diversity checks in hot evolution loops."""
    base_sig = str(genome.get_signature())
    outputs_key = tuple(
        sorted((int(cat), int(dtype), int(idx)) for cat, dtype, idx in genome.outputs)
    )
    cache_key = (base_sig, outputs_key)
    cached_key = getattr(genome, "_evolvo_eval_sig_key", None)
    if cached_key == cache_key:
        cached = getattr(genome, "_evolvo_eval_sig", None)
        if isinstance(cached, str):
            return cached

    if outputs_key:
        outs_blob = ";".join(
            f"{cat}:{dtype}:{idx}" for cat, dtype, idx in outputs_key
        )
    else:
        outs_blob = ""
    payload = f"{base_sig}|outs={outs_blob}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
    genome._evolvo_eval_sig_key = cache_key  # type: ignore[attr-defined]
    genome._evolvo_eval_sig = digest  # type: ignore[attr-defined]
    return digest


def _make_random_genome(
    initial_instructions: int,
    *,
    genome_type: str,
    slot_count: Optional[int] = None,
) -> GFSLGenome:
    genome = GFSLGenome(genome_type, slot_count=slot_count)
    for _ in range(random.randint(1, max(1, int(initial_instructions)))):
        if _append_random_instruction_fast(genome, max_attempts=6):
            continue
        try:
            genome.add_instruction_interactive(max_attempts=8)
        except RuntimeError:
            break
    genome.rebuild_validator_state()
    _invalidate_genome_caches(genome)
    return genome


class GFSLEvolver:
    """Evolution engine for GFSL genomes."""

    def __init__(
        self,
        population_size: int = 50,
        supervised_guide: Optional["GFSLSupervisedGuide"] = None,
        *,
        parent_pool_ratio: float = 0.60,
        stagnation_patience: int = 4,
        mutation_floor: float = 0.12,
        mutation_ceiling: float = 0.55,
        mutation_step: float = 0.05,
        max_instruction_count: int = 128,
        max_effective_instruction_count: int = 72,
    ):
        self.population_size = max(2, int(population_size))
        self.population: List[GFSLGenome] = []
        self.generation = 0
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7
        self.elite_ratio = 0.1
        self.diversity_cache: Set[str] = set()
        self.supervised_guide = supervised_guide

        self.parent_pool_ratio = _clamp(parent_pool_ratio, 0.25, 1.0)
        self.stagnation_patience = max(1, int(stagnation_patience))
        self.mutation_floor = _clamp(mutation_floor, 0.01, 1.0)
        self.mutation_ceiling = _clamp(
            max(self.mutation_floor, float(mutation_ceiling)),
            self.mutation_floor,
            1.0,
        )
        self.mutation_step = _clamp(mutation_step, 0.005, 0.5)
        self.max_instruction_count = max(8, int(max_instruction_count))
        self.max_effective_instruction_count = max(
            4,
            min(self.max_instruction_count, int(max_effective_instruction_count)),
        )

        self.best_fitness_tracker = -float("inf")
        self.stagnation_count = 0
        self.best_signature_tracker = ""
        self.signature_stagnation_count = 0
        self._early_stop_generation: Optional[int] = None
        self._early_stop_reason: Optional[str] = None
        self._last_stall_immigrants = 0

        self.mutation_rate = _clamp(
            max(
                float(self.mutation_floor),
                float(self.mutation_rate),
                float(self.mutation_floor) + (0.35 * float(self.mutation_step)),
            ),
            float(self.mutation_floor),
            float(self.mutation_ceiling),
        )

    def set_supervised_guide(self, guide: "GFSLSupervisedGuide"):
        """Attach or replace the supervised guidance model."""
        self.supervised_guide = guide

    def _enforce_complexity_budget(self, genome: GFSLGenome) -> None:
        max_total = max(8, int(self.max_instruction_count))
        max_effective = max(
            4,
            min(max_total, int(self.max_effective_instruction_count)),
        )
        if not genome.instructions:
            return

        total = len(genome.instructions)
        if total <= max_total:
            effective = len(genome.extract_effective_algorithm())
            if effective <= max_effective:
                return
        else:
            # Trim obvious dead weight first when genomes drift too large.
            try:
                genome.prune_stale_instructions(
                    min_hits=1,
                    include_never_used=True,
                    keep_effective=True,
                    max_pruned=max(0, total - max_total),
                )
            except Exception:
                pass

        effective_indices = list(genome.extract_effective_algorithm())
        keep_effective = (
            set(effective_indices[-max_effective:])
            if len(effective_indices) > max_effective
            else set(effective_indices)
        )

        target_keep = min(len(genome.instructions), max_total)
        keep_indices = sorted(keep_effective)
        if len(keep_indices) < target_keep:
            for idx in range(len(genome.instructions) - 1, -1, -1):
                if idx in keep_effective:
                    continue
                keep_indices.append(idx)
                if len(keep_indices) >= target_keep:
                    break

        keep_indices = sorted(set(keep_indices))
        if not keep_indices:
            keep_indices = list(range(min(len(genome.instructions), max_total)))
        if len(keep_indices) > max_total:
            keep_indices = keep_indices[-max_total:]
        if len(keep_indices) >= len(genome.instructions):
            return

        old_instructions = genome.instructions
        old_activity = genome.instruction_activity
        genome.instructions = [old_instructions[idx].copy() for idx in keep_indices]
        genome.instruction_activity = [
            copy.deepcopy(old_activity[idx])
            for idx in keep_indices
            if idx < len(old_activity)
        ]
        genome.rebuild_validator_state()

    def initialize_population(self, genome_type: str = "algorithm", initial_instructions: int = 10):
        """Create initial random population with strict diversity fallbacks."""
        self.population = []
        self.diversity_cache = set()
        attempts = 0
        max_attempts = max(self.population_size * 8, self.population_size)

        while len(self.population) < self.population_size and attempts < max_attempts:
            attempts += 1
            genome = GFSLGenome(genome_type)
            target_count = random.randint(1, max(1, int(initial_instructions)))
            for _ in range(target_count):
                if not _append_random_instruction_fast(genome, max_attempts=6):
                    break

            genome.rebuild_validator_state()
            self._enforce_complexity_budget(genome)
            _invalidate_genome_caches(genome)
            signature = _evaluation_signature(genome)
            if signature in self.diversity_cache:
                continue
            self.diversity_cache.add(signature)
            self.population.append(genome)

        while len(self.population) < self.population_size:
            genome = _make_random_genome(
                max(4, int(initial_instructions)),
                genome_type=genome_type,
            )
            self._enforce_complexity_budget(genome)
            _invalidate_genome_caches(genome)
            self.population.append(genome)

    def _stagnation_pressure(self) -> float:
        fitness_pressure = float(self.stagnation_count) / float(
            max(1, self.stagnation_patience)
        )
        signature_pressure = float(self.signature_stagnation_count) / float(
            max(1.0, float(self.stagnation_patience) * 1.5)
        )
        coupled_pressure = (0.60 * fitness_pressure) + (0.55 * signature_pressure)
        return _clamp(
            max(fitness_pressure, 1.10 * signature_pressure, coupled_pressure),
            0.0,
            1.0,
        )

    def _mutate_slot_fast(
        self,
        genome: GFSLGenome,
        *,
        diversification: float,
    ) -> bool:
        if not genome.instructions:
            return False

        instr_idx = random.randrange(len(genome.instructions))
        base_instruction = genome.instructions[instr_idx]
        slot_count = len(base_instruction.slots)
        if slot_count <= 0:
            return False

        max_trials = max(3, min(10, slot_count * 2))
        for _ in range(max_trials):
            candidate = base_instruction.copy()
            slot_idx = random.randrange(slot_count)
            valid_options = genome.validator.get_valid_options(candidate, slot_idx)
            if not valid_options:
                continue

            current_val = candidate.slots[slot_idx]
            alternatives = [opt for opt in valid_options if opt != current_val]
            if alternatives:
                candidate.slots[slot_idx] = random.choice(alternatives)
                changed = True
            else:
                changed = False
                if random.random() >= diversification:
                    continue

            valid_suffix = True
            for next_slot in range(slot_idx + 1, slot_count):
                next_valid = genome.validator.get_valid_options(candidate, next_slot)
                if not next_valid:
                    valid_suffix = False
                    break
                next_current = candidate.slots[next_slot]
                if next_current not in next_valid:
                    candidate.slots[next_slot] = random.choice(next_valid)
                    changed = True
                    continue
                if random.random() < diversification and len(next_valid) > 1:
                    next_alternatives = [opt for opt in next_valid if opt != next_current]
                    if next_alternatives:
                        candidate.slots[next_slot] = random.choice(next_alternatives)
                        changed = True

            if not valid_suffix or not changed:
                continue

            genome.instructions[instr_idx] = candidate
            return True
        return False

    def _pick_mutation_operation(self, *, instruction_count: int, pressure: float) -> str:
        operations: List[str] = ["slot", "add", "swap"]
        slot_weight = max(0.10, 0.46 - (0.16 * pressure))
        add_weight = 0.24 + (0.22 * pressure)
        if instruction_count >= int(self.max_instruction_count):
            add_weight = 0.02
        swap_weight = 0.10 + (0.10 * pressure)
        weights: List[float] = [slot_weight, add_weight, swap_weight]

        if instruction_count > 1:
            operations.append("remove")
            remove_weight = 0.14 + (0.06 * pressure)
            if instruction_count >= int(self.max_instruction_count):
                remove_weight += 0.18
            weights.append(remove_weight)

        return random.choices(operations, weights=weights, k=1)[0]

    def mutate(self, genome: GFSLGenome) -> GFSLGenome:
        """High-entropy mutation operator with complexity guards."""
        mutated = copy.deepcopy(genome)
        if not mutated.instructions:
            _append_random_instruction_fast(mutated, max_attempts=12)

        pressure = self._stagnation_pressure()
        diversification = _clamp(0.20 + (0.55 * pressure), 0.20, 0.85)
        passes = 1
        if random.random() < (0.35 + (0.45 * pressure)):
            passes += 1
        if pressure > 0.65 and random.random() < 0.80:
            passes += 1

        changed = False
        validator_synced = True
        for _ in range(passes):
            op = self._pick_mutation_operation(
                instruction_count=len(mutated.instructions),
                pressure=pressure,
            )
            if op in {"slot", "add"} and not validator_synced:
                mutated.rebuild_validator_state()
                validator_synced = True

            if op == "slot":
                changed = self._mutate_slot_fast(
                    mutated,
                    diversification=diversification,
                ) or changed
            elif op == "add":
                if len(mutated.instructions) >= int(self.max_instruction_count):
                    continue
                changed = (
                    _append_random_instruction_fast(
                        mutated,
                        max_attempts=max(6, 10 + int(6 * pressure)),
                    )
                    or changed
                )
            elif op == "swap" and len(mutated.instructions) > 1:
                a, b = random.sample(range(len(mutated.instructions)), 2)
                mutated.instructions[a], mutated.instructions[b] = (
                    mutated.instructions[b],
                    mutated.instructions[a],
                )
                if len(mutated.instruction_activity) > max(a, b):
                    mutated.instruction_activity[a], mutated.instruction_activity[b] = (
                        mutated.instruction_activity[b],
                        mutated.instruction_activity[a],
                    )
                changed = True
                validator_synced = False
            elif op == "remove" and len(mutated.instructions) > 1:
                idx = random.randrange(len(mutated.instructions))
                mutated.instructions.pop(idx)
                if idx < len(mutated.instruction_activity):
                    mutated.instruction_activity.pop(idx)
                changed = True
                validator_synced = False

        if not changed:
            if validator_synced:
                changed = self._mutate_slot_fast(
                    mutated,
                    diversification=max(diversification, 0.45),
                )
            if not changed and len(mutated.instructions) < int(self.max_instruction_count):
                changed = _append_random_instruction_fast(mutated, max_attempts=14)

        mutated.rebuild_validator_state()
        self._enforce_complexity_budget(mutated)
        if not mutated.instructions:
            _append_random_instruction_fast(mutated, max_attempts=8)
            mutated.rebuild_validator_state()

        _invalidate_genome_caches(mutated)
        mutated.fitness = None
        mutated.generation = genome.generation
        return mutated

    def crossover(self, parent1: GFSLGenome, parent2: GFSLGenome) -> GFSLGenome:
        """Crossover two genomes."""
        child = GFSLGenome(parent1.genome_type)

        if parent1.instructions and parent2.instructions:
            point1 = random.randint(0, len(parent1.instructions))
            point2 = random.randint(0, len(parent2.instructions))

            for instr in parent1.instructions[:point1]:
                child.add_instruction(instr)
            for instr in parent2.instructions[point2:]:
                child.add_instruction(instr)

        parent_outputs = random.choice([parent1.outputs, parent2.outputs])
        child.outputs = parent_outputs.copy()
        child.rebuild_validator_state()
        _invalidate_genome_caches(child)
        self._enforce_complexity_budget(child)
        return child

    def evolve(
        self,
        generations: int,
        evaluator: Callable[[GFSLGenome], float],
        progress_callback: Optional[Callable[[int, GFSLGenome, float], None]] = None,
        batch_evaluator: Optional[Callable[[List[GFSLGenome]], None]] = None,
        should_stop: Optional[Callable[..., object]] = None,
    ):
        """Main evolution loop with adaptive mutation and diversity fallback."""
        self._early_stop_generation = None
        self._early_stop_reason = None
        self._last_stall_immigrants = 0

        for gen in range(generations):
            self.generation = gen

            if batch_evaluator is not None:
                try:
                    batch_evaluator(self.population)
                except Exception:
                    for genome in self.population:
                        if genome.fitness is None:
                            try:
                                genome.fitness = evaluator(genome)
                            except Exception:
                                genome.fitness = -float("inf")
            else:
                for genome in self.population:
                    if genome.fitness is None:
                        try:
                            genome.fitness = evaluator(genome)
                        except Exception:
                            genome.fitness = -float("inf")

            self.population.sort(
                key=lambda g: g.fitness or -float("inf"),
                reverse=True,
            )
            if not self.population:
                return

            current_best = float(self.population[0].fitness or -float("inf"))
            current_signature = _evaluation_signature(self.population[0])

            if current_best > (self.best_fitness_tracker + 1e-12):
                self.best_fitness_tracker = current_best
                self.stagnation_count = 0
                self.mutation_rate = max(
                    self.mutation_floor,
                    self.mutation_rate
                    - (_MUTATION_IMPROVEMENT_DECAY_STEP_MULTIPLIER * self.mutation_step),
                )
            else:
                self.stagnation_count += 1
                if self.stagnation_count >= self.stagnation_patience:
                    signature_plateau = min(
                        1.0,
                        float(self.signature_stagnation_count)
                        / float(max(1, self.stagnation_patience)),
                    )
                    mutation_boost = float(self.mutation_step) * (
                        _MUTATION_STAGNATION_BOOST_STEP_MULTIPLIER
                        + (0.35 * signature_plateau)
                    )
                    self.mutation_rate = min(
                        self.mutation_ceiling,
                        self.mutation_rate + mutation_boost,
                    )
                    self.stagnation_count = max(0, self.stagnation_patience - 1)

            if current_signature == self.best_signature_tracker:
                self.signature_stagnation_count += 1
            else:
                self.best_signature_tracker = current_signature
                self.signature_stagnation_count = 0

            if self.signature_stagnation_count >= max(1, self.stagnation_patience):
                self.mutation_rate = min(
                    self.mutation_ceiling,
                    max(
                        self.mutation_floor,
                        self.mutation_rate
                        + max(
                            0.01,
                            _MUTATION_SIGNATURE_STALL_BOOST_STEP_MULTIPLIER
                            * float(self.mutation_step),
                        ),
                    ),
                )

            stagnation_pressure = self._stagnation_pressure()

            if self.supervised_guide:
                self.supervised_guide.observe_population(self.population)

            if progress_callback:
                best = self.population[0]
                progress_callback(gen, best, best.fitness or -float("inf"))

            if should_stop is not None:
                stop = False
                stop_reason = "adaptive-stagnation"
                try:
                    decision = should_stop(
                        gen,
                        self.population[0],
                        float(self.population[0].fitness or -float("inf")),
                        self.population,
                    )
                    if isinstance(decision, tuple):
                        stop = bool(decision[0]) if len(decision) > 0 else False
                        if len(decision) > 1 and str(decision[1]).strip():
                            stop_reason = str(decision[1]).strip()
                    elif isinstance(decision, str):
                        stop = bool(decision)
                        if decision.strip():
                            stop_reason = decision.strip()
                    else:
                        stop = bool(decision)
                except Exception:
                    stop = False

                if stop:
                    self._early_stop_generation = int(gen)
                    self._early_stop_reason = str(stop_reason)
                    break

            if gen % 10 == 0:
                best = self.population[0]
                effective_size = len(best.extract_effective_algorithm())
                print(
                    f"Gen {gen:03d}: Best Fitness={best.fitness:.4f}, "
                    f"Effective Size={effective_size}/{len(best.instructions)}, "
                    f"MutRate={self.mutation_rate:.3f}"
                )

            elite_size = max(1, int(self.population_size * self.elite_ratio))
            new_population = self.population[:elite_size]

            seen = {_evaluation_signature(genome) for genome in new_population}
            attempts = 0
            max_attempts = max(self.population_size * 90, 240)

            parent_pool_size = max(2, int(len(self.population) * self.parent_pool_ratio))
            parent_pool = self.population[:parent_pool_size]
            local_refine_chance = _clamp(0.34 - (0.26 * stagnation_pressure), 0.05, 0.34)

            guide_sizes = [
                max(1, len(genome.extract_effective_algorithm()))
                for genome in parent_pool[: max(4, min(len(parent_pool), elite_size + 2))]
            ]
            guide_mean = (
                (sum(guide_sizes) / float(len(guide_sizes)))
                if guide_sizes
                else 8.0
            )
            immigrant_instruction_budget = max(
                4,
                min(24, int(round((guide_mean * 1.55) + 2.0))),
            )
            fallback_instruction_budget = max(
                4,
                min(20, int(round((guide_mean * 1.20) + 1.0))),
            )

            def tournament_from(pool: Sequence[GFSLGenome], size: int = 3) -> GFSLGenome:
                tournament = random.sample(list(pool), min(size, len(pool)))
                return max(tournament, key=lambda g: g.fitness or -float("inf"))

            while len(new_population) < self.population_size and attempts < max_attempts:
                attempts += 1
                if stagnation_pressure > 0.0 and random.random() < local_refine_chance:
                    parent = parent_pool[attempts % len(parent_pool)]
                    child = copy.deepcopy(parent)
                    child = self.mutate(child)
                else:
                    parent1 = tournament_from(parent_pool)
                    parent2 = tournament_from(parent_pool)
                    if random.random() < self.crossover_rate:
                        child = self.crossover(parent1, parent2)
                    else:
                        child = copy.deepcopy(random.choice([parent1, parent2]))

                child.fitness = None
                child.generation = gen + 1
                self._enforce_complexity_budget(child)

                force_mutation = (
                    stagnation_pressure >= float(_MUTATION_FORCE_THRESHOLD)
                    and random.random() < float(_MUTATION_FORCE_CHANCE)
                )
                if force_mutation or random.random() < self.mutation_rate:
                    if self.supervised_guide:
                        child = self.supervised_guide.propose_mutation(self, child)
                    else:
                        child = self.mutate(child)
                    child.fitness = None
                    child.generation = gen + 1
                    self._enforce_complexity_budget(child)

                signature = _evaluation_signature(child)
                if signature in seen:
                    continue
                seen.add(signature)
                new_population.append(child)

            if self.signature_stagnation_count >= max(1, self.stagnation_patience):
                immigrant_ratio = 0.22 + (0.33 * stagnation_pressure)
                immigrant_target = max(1, int(round(self.population_size * immigrant_ratio)))
                injected = 0
                inject_attempts = 0
                preserve = max(1, int(elite_size))
                while (
                    injected < immigrant_target
                    and inject_attempts < (immigrant_target * 12)
                ):
                    inject_attempts += 1
                    immigrant = _make_random_genome(
                        immigrant_instruction_budget,
                        genome_type=self.population[0].genome_type,
                    )
                    immigrant.fitness = None
                    immigrant.generation = gen + 1
                    signature = _evaluation_signature(immigrant)
                    if signature in seen:
                        continue

                    if len(new_population) >= self.population_size:
                        replace_idx = len(new_population) - 1 - injected
                        if replace_idx < preserve:
                            break
                        replaced_sig = _evaluation_signature(new_population[replace_idx])
                        seen.discard(replaced_sig)
                        new_population[replace_idx] = immigrant
                    else:
                        new_population.append(immigrant)

                    seen.add(signature)
                    injected += 1

                self._last_stall_immigrants = int(injected)

            while len(new_population) < self.population_size:
                fallback = copy.deepcopy(random.choice(self.population[:elite_size]))
                fallback.fitness = None
                fallback.generation = gen + 1
                fallback_added = False

                for _ in range(16):
                    signature = _evaluation_signature(fallback)
                    if signature not in seen:
                        seen.add(signature)
                        new_population.append(fallback)
                        fallback_added = True
                        break
                    fallback = self.mutate(copy.deepcopy(fallback))
                    fallback.fitness = None
                    fallback.generation = gen + 1

                if fallback_added:
                    continue

                immigrant = _make_random_genome(
                    fallback_instruction_budget,
                    genome_type=self.population[0].genome_type,
                )
                immigrant.fitness = None
                immigrant.generation = gen + 1
                signature = _evaluation_signature(immigrant)
                if signature in seen:
                    continue
                seen.add(signature)
                new_population.append(immigrant)

            self.population = new_population[: self.population_size]
            self.diversity_cache = set(seen)

    def _tournament_select(self, tournament_size: int = 3) -> GFSLGenome:
        """Tournament selection from the current population."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness or -float("inf"))


__all__ = ["GFSLEvolver"]
