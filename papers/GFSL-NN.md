# GFSL-NN: fixed-structure genetic description of neural architectures

**Status:** implemented in `src/evolvo/nnarch/` (draft 0.1, 17 November 2026).
**Parent specification:** [GFSL definition](GFSL-definition.md).
**Purpose:** describe dynamically generated PyTorch models with a genome that is valid by
construction, search it with weights learned while searching, and train each candidate only
as long as training still pays.

## 1. Design goals

1. **Every genome builds.** Options are enumerated per slot and filtered by what the earlier
   slots and the available tensors allow. There are no free-form numbers, so there are no shape errors.
2. **No fixed preference at implementation level.** Which ops, sizes and edits tend to work is
   learned from rewards while the search runs, per *context*. The contexts that matter
   are discovered from the data, not enumerated.
3. **Compact repetition.** Repeated compositions (MLP stacks, residual blocks, transformer
   encoders, DeepSets) are macros. They are functions with enumerated arguments, so "depth" is one
   integer gene.
4. **Efficiency is part of fitness.** Size is penalized in a scale-free way, a learned
   loss-versus-size frontier steers growth, and each candidate stops training at its plateau.

## 2. Rows

A genome is an ordered list of rows. Each row has the fixed GFSL shape of seven integer slots:

```text
[ target | op | src_a | src_b | arg0 | arg1 | arg2 ]
```

- `target` is implicit and SSA-like: row *i* writes value `len(inputs) + i`. Values are never
  overwritten, so the genome is a DAG.
- `op` indexes an append-only registry of primitives and macros (codes stay stable).
- `src_a`, `src_b` are earlier values (`-1` = NONE for unary ops).
- `arg*` are indices into the op's per-argument *option list* at that point. Unused args are `-1`.

The human-readable form (`ArchSpace.describe`) is not a wire format:

```text
v0 = input vec[4]
v1 = tokenize(v0, dim=16) -> set[16]@n4
v2 = transformer(v1, depth=2, heads=4, ff=2) -> set[16]@n4
v3 = pool(v2, mode=mean) -> vec[16]
out = linear(v3 -> 1)
```

## 3. Types

| Kind | Tensor | Notes |
| --- | --- | --- |
| `vec` | `[batch, width]` | tabular / pooled |
| `set` | `[batch, n, width]` | members, tokens, sequence; `origin` names `n` |

Element-wise binary ops (`add`, `mul`, `concat`, `context`) only pair sets with the same origin.
The head is fixed, not genetic: a `set` result is mean-pooled, then a linear layer maps to the
output width.

## 4. Operations

| Primitive | Sources | Arguments | Result |
| --- | --- | --- | --- |
| `linear` | any | `units` (width ladder) | same kind, `units` |
| `act` | any | `fn` (space activations) | same |
| `norm` | any | — | same (LayerNorm) |
| `dropout` | any | `rate` | same |
| `add`, `mul` | two equal shapes | — | same |
| `concat` | two of one kind/origin | — | widths summed |
| `attend` | set | `heads` (divisors of width) | same (self-attention) |
| `pool` | set | `mode` (mean/sum/max) | vec |
| `context` | set + vec | — | set, vec appended per member |
| `tokenize` | vec | `dim` (width ladder) | set of one token per feature |

| Macro | Arguments | Expansion |
| --- | --- | --- |
| `mlp` | `units, depth, act` | depth × (linear → act) |
| `resmlp` | `depth, ratio, act` | depth × (x + linear(act(linear(norm x, w·ratio)), w)) |
| `transformer` | `depth, heads, ff` (set) | depth × pre-norm attention block + feed-forward block |
| `deepsets` | `units, depth, pool` (set) | `mlp` per member, then `pool` |

Macros expand through an `Emitter` (`emitter.emit(op, *sources, **args)`). They may call other
macros, up to `max_macro_nesting`. `register_macro(registry, name, args, expand)` adds a user
macro, which is then sampled, mutated and validated like any primitive.

## 5. Cascading validity

For row *i* with the values (and shapes) before it, the cascade is:

1. **op**: ops for which some source, second source and cheapest argument set yields a row
   within the parameter budget;
2. **src_a**: values the op accepts that still admit such a row;
3. **src_b**: values matching `src_a` under the op's pairing rule;
4. **args**, in order: enumerated options for which the row, with later arguments at their
   cheapest option, keeps `parameters so far + row + head ≤ max_params`. Heads are offered only
   when they divide the width.

`ArchSpace.repair` replays this cascade over any row list. It keeps each choice that is still
valid. A numeric choice that became invalid moves to the *nearest* valid option on a log scale,
so intent is preserved. Anything else is re-sampled. Mutation and crossover produce raw edits
and go through repair, which is why every evaluated genome is valid by construction.
`ArchSpace.validate` lists every reason a genome is not in the language.

## 6. Values and the adaptive ladder

Widths come from a geometric ladder with ratio √2 (4, 6, 8, 11, 16, 23, 32, 45, 64, …) inside
`width_bounds`. Only a *window* of it is active. The window starts around four times the
problem's widest input or output and widens by two rungs whenever an elite genome uses a width
at its edge. Sizes therefore grow where the search finds evidence, not where an implementer
guessed. Depths are `1 … max_depth`. Activations, dropout rates, pools, ratios and head counts
are per-space lists.

## 7. Learned choice weights

`AdaptivePolicy` samples every slot (also the genome length and the mutation operator) with
probability

```text
p(option) = (1 − ε) · softmax((r̂(option | context) + bonus) / τ) + ε / n
```

where `r̂` is a `ContextualStats` estimate of the reward of genomes that made that choice, and
the reward is the fitness quantile among all evaluations (1 = best). The bonus is an optimism
term that decays with the option's own count. Caller-supplied *prior logits* tilt ascending
size options by the efficiency pressure (§9).

### Dynamic group types

A context is a mapping of features: slot, op, source kind, width bucket, row position,
remaining budget bucket, and the task signature (input width, output width, input kinds). Which
combinations behave alike is not declared. `ContextualStats`:

- orders the features greedily by *adjusted* extra between-group variance (gain minus the
  chance gain of the groups it adds) and re-learns that order as data grows;
- aggregates every prefix of the order, so groups are the cells of a learned hierarchy;
- estimates by shrinking each level toward its parent with `κ` pseudo-observations
  (empirical-Bayes backoff). Sparse specific groups borrow from general ones.

The same store learns epoch tendencies (§8). Stores serialize to JSON as part of a `Knowledge`
file, so later searches on similar tasks start from what earlier ones learned.

## 8. Training: automatic epoch stopping

`PlateauStopper` ends a candidate's training on:

- `plateau`: no relative improvement > `rel_tol` for `max(patience, relative_patience × epochs)`,
  measured from the best loss at the last counted improvement (slow steady gains add up);
- `hopeless`: after `hopeless_min_epochs`, the run is behind the incumbent at equal effort (its
  best after *t* epochs is worse than the incumbent's best after *t* epochs by a margin) and
  worse than the incumbent's final loss, and its fitted learning curve `a + b·t^−α` does not
  reach that loss;
- `diverged` or the hard `max_epochs`.

The patience is learned. The stopper reads the `gap` (epochs between improvements) tendency
for its context (task signature, size bucket, effective row count) and records its own gaps and
best epoch afterwards. A learned value never stops a run that is still improving. Weights are
restored to the best validation epoch.

## 9. Efficiency of structure

Fitness is `score = ln(loss) + c · ln(params)`. A doubling of size pays when it cuts the loss by
more than `1 − 2^−c`. `EfficiencyModel` keeps the best loss per half-octave of size (the
achievable frontier) and fits `L(P) = L∞ + B·P^−α` with relative-error weighting. From that fit
it derives:

- `efficient_size()`: the P* where the gain per doubling falls to `1 − 2^−c`;
- `pressure(P) ∈ [−1, 1]`: positive below P* (grow), negative above (shrink).

The pressure biases size-argument sampling and the direction of argument-step mutations. Runs
stopped as hopeless are censored and kept out of the frontier.

## 10. Search

`ArchitectureSearch` evaluates a seeded or random generation. After that each generation
keeps `elite` genomes and fills the rest by tournament selection plus one learned mutation:

- `arg` — step a numeric argument along its ladder, or re-choose a categorical one;
- `replace` — a row;
- `insert`, `delete` — with bypass rewiring;
- `rewire` — a source;
- `crossover` — prefix of one parent + suffix of another, references remapped.

Every edit goes through repair. Effective signatures are cached. Introns are kept, for
neutral drift. The search stops when the best score has not improved for `patience`
generations (`converged`), or on generation, evaluation or time budgets. The report includes
the best genome (readable form, slots, lowered graph), the Pareto front, the efficiency fit, the
learned feature orders and groups, and the mutation preferences.

## 11. Limits (draft 0.1)

- `set` inputs are dense (no padding mask). Masked sets are future work.
- Only the macros listed are built in. Mining new macros from recurring elite fragments is
  future work.
- The search is single-objective with a size penalty. FLOP/latency costs and multi-fidelity
  schedules beyond hopeless stopping are not implemented.
