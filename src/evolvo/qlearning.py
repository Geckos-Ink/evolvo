"""Q-learning guide for GFSL instruction construction."""

from __future__ import annotations

import copy
import itertools
import math
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Tuple

from .instruction import GFSLInstruction
from .slots import DEFAULT_SLOT_COUNT
from .validator import SlotValidator


class GFSLQLearningGuide:
    """
    Q-Learning agent for guiding GFSL instruction construction.
    Learns which slot values work best given previous slots.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        epsilon: float = 0.1,
        *,
        replace_probability: float = 0.5,
        top_fraction: float = 0.10,
        reward_clip: float = 5.0,
        use_viability: bool = False,
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = 0.95
        self.replace_probability = float(replace_probability)
        self.top_fraction = float(top_fraction)
        self.reward_clip = float(reward_clip)
        self.use_viability = bool(use_viability)  # evolver path: fast cascading options

        self.q_table = defaultdict(lambda: defaultdict(float))
        self.experience_buffer = deque(maxlen=1000)
        self._pending: Dict[int, List[Tuple[str, int]]] = {}
        self.observations = 0
        self.updates = 0

    def get_state_key(self, instruction: GFSLInstruction, slot_index: int) -> str:
        """Generate state key from previous slots."""
        prev_slots = instruction.slots[:slot_index]
        return f"{slot_index}:{':'.join(str(s) for s in prev_slots)}"

    def choose_action(
        self, instruction: GFSLInstruction, slot_index: int, valid_options: List[int]
    ) -> int:
        """Choose slot value using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(valid_options)

        state_key = self.get_state_key(instruction, slot_index)
        q_values = self.q_table[state_key]

        valid_q = {opt: q_values.get(opt, 0.0) for opt in valid_options}

        if not valid_q:
            return random.choice(valid_options)

        max_q = max(valid_q.values())
        best_actions = [a for a, q in valid_q.items() if q == max_q]
        return random.choice(best_actions)

    def build_instruction(
        self, validator: SlotValidator, max_attempts: int = 25, *, use_viability: bool = True
    ) -> Tuple[GFSLInstruction, List[Tuple[str, int]]]:
        """Build instruction using Q-learning guidance.

        ``use_viability`` restricts each slot to options that can still complete the instruction
        (``SlotValidator.viable_options``, which builds probability trees and is slow); with
        ``False`` the plain cascading options are used and a dead end simply retries.
        """
        if validator.slot_count == 0:
            validator.slot_count = DEFAULT_SLOT_COUNT
        last_error: Optional[Exception] = None
        for _ in range(max_attempts):
            instruction = GFSLInstruction(slot_count=validator.slot_count)
            states_actions: List[Tuple[str, int]] = []
            try:
                for slot_idx in range(validator.slot_count):
                    valid_options = validator.get_valid_options(instruction, slot_idx)
                    if not valid_options:
                        raise ValueError(f"No valid options for slot {slot_idx}.")

                    if use_viability:
                        viable, _ = validator.viable_options(instruction, slot_idx)
                        action_space = viable or valid_options
                    else:
                        action_space = valid_options

                    state_key = self.get_state_key(instruction, slot_idx)
                    action = self.choose_action(instruction, slot_idx, action_space)

                    instruction.slots[slot_idx] = action
                    states_actions.append((state_key, action))
                return instruction, states_actions
            except ValueError as exc:
                last_error = exc
                continue

        raise RuntimeError(
            f"Unable to build a valid instruction after {max_attempts} attempts."
        ) from last_error

    def update_from_fitness(self, states_actions: List[Tuple[str, int]], fitness: float):
        """Update Q-values based on fitness reward."""
        reward = fitness

        for i in range(len(states_actions) - 1, -1, -1):
            state_key, action = states_actions[i]

            current_q = self.q_table[state_key][action]

            if i == len(states_actions) - 1:
                new_q = current_q + self.learning_rate * (reward - current_q)
            else:
                next_state, next_action = states_actions[i + 1]
                next_q = self.q_table[next_state][next_action]
                new_q = current_q + self.learning_rate * (
                    reward + self.discount_factor * next_q - current_q
                )

            self.q_table[state_key][action] = new_q

    # ------------------------------------------------------------------ evolver interface
    # Same plug-in protocol as GFSLSupervisedGuide / GFSLPolicyGuide:
    # GFSLEvolver(supervised_guide=GFSLQLearningGuide()).

    def propose_mutation(self, evolver, genome):
        """Child = parent with one Q-learning-built instruction replacing a random instruction
        (probability ``replace_probability``) or appended."""
        from .evolver import _invalidate_genome_caches

        child = copy.deepcopy(genome)
        child.rebuild_validator_state()
        try:
            instruction, states_actions = self.build_instruction(
                child.validator, use_viability=self.use_viability
            )
        except RuntimeError:
            return evolver.mutate(genome)
        if child.instructions and (
            random.random() < self.replace_probability
            or len(child.instructions) >= int(getattr(evolver, "max_instruction_count", 128))
        ):
            idx = random.randrange(len(child.instructions))
            child.instructions[idx] = instruction
        else:
            child.instructions.append(instruction)
        child.rebuild_validator_state()
        evolver._enforce_complexity_budget(child)
        _invalidate_genome_caches(child)
        child.fitness = None
        child.generation = genome.generation
        token = next(_TOKENS)
        child._qlearning_token = token  # type: ignore[attr-defined]
        self._pending[token] = states_actions
        return child

    def observe_population(self, population: Sequence) -> None:
        """Update Q-values from the children evaluated since the last call.

        Reward = ``(f - q) / (|q| + 1e-8)`` clipped to ``±reward_clip``, with ``q`` the mean
        fitness of the population's top ``top_fraction`` (same generation, same scoring).
        """
        finite = sorted(
            (float(g.fitness) for g in population if g.fitness is not None and math.isfinite(float(g.fitness))),
            reverse=True,
        )
        if finite:
            k = max(1, int(math.ceil(self.top_fraction * len(finite))))
            q = sum(finite[:k]) / k
            for genome in population:
                token = getattr(genome, "_qlearning_token", None)
                states_actions = self._pending.pop(token, None) if token is not None else None
                if states_actions is None or genome.fitness is None:
                    continue
                f = float(genome.fitness)
                if math.isfinite(f):
                    reward = max(-self.reward_clip, min(self.reward_clip, (f - q) / (abs(q) + 1e-8)))
                else:
                    reward = -self.reward_clip
                self.update_from_fitness(states_actions, reward)
                self.updates += 1
        self._pending.clear()
        self.observations += 1


_TOKENS = itertools.count(1)


__all__ = ["GFSLQLearningGuide"]
