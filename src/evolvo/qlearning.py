"""Q-learning guide for GFSL instruction construction."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import List, Optional, Tuple

from .instruction import GFSLInstruction
from .slots import DEFAULT_SLOT_COUNT
from .validator import SlotValidator


class GFSLQLearningGuide:
    """
    Q-Learning agent for guiding GFSL instruction construction.
    Learns which slot values work best given previous slots.
    """

    def __init__(self, learning_rate: float = 0.1, epsilon: float = 0.1):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = 0.95

        self.q_table = defaultdict(lambda: defaultdict(float))
        self.experience_buffer = deque(maxlen=1000)

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
        self, validator: SlotValidator, max_attempts: int = 25
    ) -> Tuple[GFSLInstruction, List[Tuple[str, int]]]:
        """Build instruction using Q-learning guidance."""
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

                    viable, _ = validator.viable_options(instruction, slot_idx)
                    action_space = viable or valid_options

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


__all__ = ["GFSLQLearningGuide"]
