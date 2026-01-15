"""Evolution engine for GFSL genomes."""

from __future__ import annotations

import copy
import random
from typing import Callable, List, Optional, Set, TYPE_CHECKING

from .genome import GFSLGenome

if TYPE_CHECKING:
    from .supervised import GFSLSupervisedGuide


class GFSLEvolver:
    """Evolution engine for GFSL genomes."""

    def __init__(self, population_size: int = 50, supervised_guide: Optional["GFSLSupervisedGuide"] = None):
        self.population_size = population_size
        self.population: List[GFSLGenome] = []
        self.generation = 0
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7
        self.elite_ratio = 0.1
        self.diversity_cache: Set[str] = set()
        self.supervised_guide = supervised_guide

    def set_supervised_guide(self, guide: "GFSLSupervisedGuide"):
        """Attach or replace the supervised guidance model."""
        self.supervised_guide = guide

    def initialize_population(self, genome_type: str = "algorithm", initial_instructions: int = 10):
        """Create initial random population."""
        self.population = []
        self.diversity_cache = set()
        attempts = 0
        max_attempts = self.population_size * 10

        while len(self.population) < self.population_size and attempts < max_attempts:
            genome = GFSLGenome(genome_type)

            for _ in range(random.randint(1, initial_instructions)):
                try:
                    genome.add_instruction_interactive()
                except RuntimeError:
                    break

            sig = genome.get_signature()
            if sig not in self.diversity_cache:
                self.diversity_cache.add(sig)
                self.population.append(genome)

            attempts += 1

    def mutate(self, genome: GFSLGenome) -> GFSLGenome:
        """Mutate genome using slot-level mutations."""
        mutated = copy.deepcopy(genome)
        mutation_type = random.choice(["slot", "add", "remove"])

        if mutation_type == "slot" and mutated.instructions:
            instr_idx = random.randint(0, len(mutated.instructions) - 1)
            slot_idx = random.randint(0, mutated.validator.slot_count - 1)

            test_instr = mutated.instructions[instr_idx].copy()
            valid_options = mutated.validator.get_valid_options(test_instr, slot_idx)

            if valid_options:
                current_val = test_instr.slots[slot_idx]
                other_options = [v for v in valid_options if v != current_val]
                if other_options:
                    mutated.instructions[instr_idx].slots[slot_idx] = random.choice(other_options)

                    for next_slot in range(slot_idx + 1, mutated.validator.slot_count):
                        next_valid = mutated.validator.get_valid_options(
                            mutated.instructions[instr_idx], next_slot
                        )
                        if next_valid:
                            mutated.instructions[instr_idx].slots[next_slot] = random.choice(next_valid)

        elif mutation_type == "add":
            mutated.add_instruction_interactive()

        elif mutation_type == "remove" and len(mutated.instructions) > 1:
            idx = random.randint(0, len(mutated.instructions) - 1)
            mutated.instructions.pop(idx)

        mutated._signature = None
        mutated._effective_instructions = None
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

        return child

    def evolve(
        self,
        generations: int,
        evaluator: Callable[[GFSLGenome], float],
        progress_callback: Optional[Callable[[int, GFSLGenome, float], None]] = None,
    ):
        """
        Main evolution loop.

        Args:
            generations: Number of generations to evolve
            evaluator: Fitness function
            progress_callback: Optional callback(generation, best_genome, best_fitness)
        """
        for gen in range(generations):
            self.generation = gen

            for genome in self.population:
                if genome.fitness is None:
                    try:
                        genome.fitness = evaluator(genome)
                    except Exception:
                        genome.fitness = -float("inf")

            self.population.sort(
                key=lambda g: g.fitness or -float("inf"), reverse=True
            )

            if self.supervised_guide:
                self.supervised_guide.observe_population(self.population)

            if progress_callback and self.population:
                best = self.population[0]
                progress_callback(gen, best, best.fitness or -float("inf"))

            if self.population and gen % 10 == 0:
                best = self.population[0]
                effective_size = len(best.extract_effective_algorithm())
                print(
                    f"Gen {gen:03d}: Best Fitness={best.fitness:.4f}, "
                    f"Effective Size={effective_size}/{len(best.instructions)}"
                )

            elite_size = int(self.population_size * self.elite_ratio)
            new_population = self.population[:elite_size]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()

                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(random.choice([parent1, parent2]))

                child.fitness = None
                child.generation = gen + 1

                if random.random() < self.mutation_rate:
                    if self.supervised_guide:
                        child = self.supervised_guide.propose_mutation(self, child)
                    else:
                        child = self.mutate(child)
                    child.fitness = None
                    child.generation = gen + 1

                sig = child.get_signature()
                if sig not in self.diversity_cache:
                    self.diversity_cache.add(sig)
                    new_population.append(child)

            self.population = new_population[: self.population_size]

    def _tournament_select(self, tournament_size: int = 3) -> GFSLGenome:
        """Tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness or -float("inf"))


__all__ = ["GFSLEvolver"]
