"""GFSL-NN: a fixed-row genetic language for neural architectures.

The language follows the GFSL principles (``papers/GFSL-definition.md``):

* **Fixed row shape.** Every row is ``[target, op, src_a, src_b, arg0, arg1, arg2]`` (7 integer
  slots, the GFSL default). ``target`` is implicit and SSA-like: row *i* always writes value
  ``len(inputs) + i``, so a value is never overwritten and the genome is a DAG.
* **Cascading validity.** The options offered at each slot depend on the slots before it and on
  the shapes already available: an op is offered only if some source fits it, a second source
  only if it matches the first, an argument only if it keeps the architecture inside the
  parameter budget, attention heads only if they divide the width. Sampling, mutation and
  crossover all go through :meth:`ArchSpace.repair`, so every genome they produce is valid by
  construction and builds without a shape error.
* **Enumerated values.** Arguments are chosen from finite option lists (VALUE slots). Widths come
  from a geometric ladder whose *active window* adapts during search (it widens when elite
  architectures press against an edge), so sizes are learned on the road rather than fixed at
  implementation time.
* **Macros.** Repeated compositions are ops too: ``mlp(units, depth, act)``,
  ``resmlp(depth, ratio, act)``, ``transformer(depth, heads, ff)``, ``deepsets(units, depth,
  pool)``. A macro row is one gene whose integer arguments (e.g. ``depth``) control how many
  primitive layers it expands to, so mutating a depth is a single-slot change. Macros expand
  through an :class:`Emitter`, may call other macros, and users can register their own.
* **Effective extraction.** Rows that do not reach the output are introns: kept in the genome
  (neutral drift), excluded from the built model and from the cache signature.

Tensor kinds: ``vec`` is ``[batch, width]``; ``set`` is ``[batch, n, width]`` (members, tokens or
a sequence). Set values carry an ``origin`` naming their length, so element-wise combinations
only pair sets of the same length. The output head (an automatic mean pool for a ``set`` result,
then a linear layer to the output width) is fixed, not genetic.

Pure Python: shapes and parameter counts are computed without torch. ``evolvo.nnarch.build``
lowers a genome to a ``torch.nn.Module``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

from .stats import log_bucket

VEC = "vec"
SET = "set"
NONE = -1
SLOT_COUNT = 7
MAX_ARGS = SLOT_COUNT - 4

ACTIVATIONS = ("relu", "silu", "gelu", "tanh", "sigmoid")


@dataclass(frozen=True)
class Shape:
    kind: str
    width: int
    origin: str = ""

    def describe(self) -> str:
        return f"{self.kind}[{self.width}]" + (f"@{self.origin}" if self.kind == SET else "")


@dataclass(frozen=True)
class ArgSpec:
    """One enumerated argument.

    ``domain`` names how the option list is produced:

    * ``width``  – the space's adaptive width ladder (ascending);
    * ``depth``  – ``1 .. space.max_depth``;
    * ``choice`` – the fixed ``choices``;
    * ``divisor`` – the ``choices`` that divide the first source's width (attention heads);
    * ``space:<name>`` – a list the space provides (``activations``, ``dropout``, ``pools``,
      ``ratios``, ``heads``), so a caller can restrict it per problem.

    Numeric option lists are ascending, so the first option is the cheapest.
    """

    name: str
    domain: str
    choices: Tuple[Any, ...] = ()
    numeric: bool = True


@dataclass
class OpSpec:
    """An operation of the language: a primitive or a macro."""

    name: str
    arity: int
    args: Tuple[ArgSpec, ...] = ()
    first: Callable[[Shape], bool] = lambda s: True
    second: Optional[Callable[[Shape, Shape], bool]] = None
    infer: Optional[Callable[[Tuple[Shape, ...], Dict[str, Any]], Shape]] = None
    params: Optional[Callable[[Tuple[Shape, ...], Dict[str, Any]], int]] = None
    expand: Optional[Callable[["Emitter", Tuple[int, ...], Dict[str, Any]], int]] = None
    doc: str = ""

    @property
    def is_macro(self) -> bool:
        return self.expand is not None

    def validate(self) -> None:
        if self.arity not in (1, 2):
            raise ValueError(f"op {self.name}: arity must be 1 or 2")
        if len(self.args) > MAX_ARGS:
            raise ValueError(f"op {self.name}: at most {MAX_ARGS} arguments fit a fixed row")
        if self.arity == 2 and self.second is None:
            raise ValueError(f"op {self.name}: a binary op needs a second-source rule")
        if self.is_macro == (self.infer is not None):
            raise ValueError(f"op {self.name}: give either infer/params (primitive) or expand (macro)")


class OpRegistry:
    """Ordered, append-only op table; op codes are positions, so they stay stable."""

    def __init__(self) -> None:
        self._ops: Dict[str, OpSpec] = {}

    def register(self, spec: OpSpec) -> OpSpec:
        spec.validate()
        if spec.name in self._ops:
            raise ValueError(f"op {spec.name} is already registered")
        self._ops[spec.name] = spec
        return spec

    def get(self, name: str) -> OpSpec:
        return self._ops[name]

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    def names(self) -> List[str]:
        return list(self._ops)

    def code(self, name: str) -> int:
        return self.names().index(name)


# ---------------------------------------------------------------------------- primitives
def _same(a: Shape, b: Shape) -> bool:
    return a == b


def _same_kind(a: Shape, b: Shape) -> bool:
    return a.kind == b.kind and (a.kind == VEC or a.origin == b.origin)


def _register_primitives(reg: OpRegistry) -> None:
    keep = lambda shapes, args: shapes[0]  # noqa: E731
    zero = lambda shapes, args: 0  # noqa: E731
    reg.register(OpSpec(
        "linear", 1, (ArgSpec("units", "width"),),
        infer=lambda s, a: Shape(s[0].kind, a["units"], s[0].origin),
        params=lambda s, a: s[0].width * a["units"] + a["units"],
        doc="affine map of the last axis"))
    reg.register(OpSpec(
        "act", 1, (ArgSpec("fn", "space:activations", numeric=False),),
        infer=keep, params=zero, doc="element-wise activation"))
    reg.register(OpSpec(
        "norm", 1, (), infer=keep, params=lambda s, a: 2 * s[0].width,
        doc="layer normalization of the last axis"))
    reg.register(OpSpec(
        "dropout", 1, (ArgSpec("rate", "space:dropout"),), infer=keep, params=zero,
        doc="training-time dropout (identity at inference)"))
    reg.register(OpSpec(
        "add", 2, (), second=_same, infer=keep, params=zero, doc="residual sum"))
    reg.register(OpSpec(
        "mul", 2, (), second=_same, infer=keep, params=zero, doc="element-wise gate"))
    reg.register(OpSpec(
        "concat", 2, (), second=_same_kind,
        infer=lambda s, a: Shape(s[0].kind, s[0].width + s[1].width, s[0].origin),
        params=zero, doc="concatenate the last axis"))
    reg.register(OpSpec(
        "attend", 1, (ArgSpec("heads", "divisor", numeric=True),),
        first=lambda s: s.kind == SET, infer=keep,
        params=lambda s, a: 4 * s[0].width * s[0].width + 4 * s[0].width,
        doc="multi-head self-attention across set members"))
    reg.register(OpSpec(
        "pool", 1, (ArgSpec("mode", "space:pools", numeric=False),),
        first=lambda s: s.kind == SET, infer=lambda s, a: Shape(VEC, s[0].width),
        params=zero, doc="permutation-invariant set reduction"))
    reg.register(OpSpec(
        "context", 2, (), first=lambda s: s.kind == SET, second=lambda a, b: b.kind == VEC,
        infer=lambda s, a: Shape(SET, s[0].width + s[1].width, s[0].origin),
        params=zero, doc="append a vector to every set member"))
    reg.register(OpSpec(
        "tokenize", 1, (ArgSpec("dim", "width"),), first=lambda s: s.kind == VEC,
        infer=lambda s, a: Shape(SET, a["dim"], f"n{s[0].width}"),
        params=lambda s, a: 2 * s[0].width * a["dim"],
        doc="one learned embedding per input feature (feature tokens)"))


# ---------------------------------------------------------------------------- macros
def _mlp(em: "Emitter", src: Tuple[int, ...], a: Dict[str, Any]) -> int:
    x = src[0]
    for _ in range(a["depth"]):
        x = em.emit("act", em.emit("linear", x, units=a["units"]), fn=a["act"])
    return x


def _resmlp(em: "Emitter", src: Tuple[int, ...], a: Dict[str, Any]) -> int:
    x = src[0]
    width = em.shape(x).width
    for _ in range(a["depth"]):
        h = em.emit("norm", x)
        h = em.emit("linear", h, units=width * a["ratio"])
        h = em.emit("act", h, fn=a["act"])
        h = em.emit("linear", h, units=width)
        x = em.emit("add", x, h)
    return x


def _transformer(em: "Emitter", src: Tuple[int, ...], a: Dict[str, Any]) -> int:
    x = src[0]
    width = em.shape(x).width
    for _ in range(a["depth"]):
        h = em.emit("attend", em.emit("norm", x), heads=a["heads"])
        x = em.emit("add", x, h)
        h = em.emit("linear", em.emit("norm", x), units=width * a["ff"])
        h = em.emit("linear", em.emit("act", h, fn="gelu"), units=width)
        x = em.emit("add", x, h)
    return x


def _deepsets(em: "Emitter", src: Tuple[int, ...], a: Dict[str, Any]) -> int:
    x = em.emit("mlp", src[0], units=a["units"], depth=a["depth"], act="silu")
    return em.emit("pool", x, mode=a["pool"])


def _register_macros(reg: OpRegistry) -> None:
    reg.register(OpSpec(
        "mlp", 1, (ArgSpec("units", "width"), ArgSpec("depth", "depth"),
                   ArgSpec("act", "space:activations", numeric=False)),
        expand=_mlp, doc="depth x (linear(units) -> act)"))
    reg.register(OpSpec(
        "resmlp", 1, (ArgSpec("depth", "depth"), ArgSpec("ratio", "space:ratios"),
                      ArgSpec("act", "space:activations", numeric=False)),
        expand=_resmlp, doc="depth x pre-norm residual MLP block (width preserved)"))
    reg.register(OpSpec(
        "transformer", 1, (ArgSpec("depth", "depth"), ArgSpec("heads", "divisor"),
                           ArgSpec("ff", "space:ratios")),
        first=lambda s: s.kind == SET,
        expand=_transformer, doc="depth x pre-norm encoder block (attention + feed-forward)"))
    reg.register(OpSpec(
        "deepsets", 1, (ArgSpec("units", "width"), ArgSpec("depth", "depth"),
                        ArgSpec("pool", "space:pools", numeric=False)),
        first=lambda s: s.kind == SET,
        expand=_deepsets, doc="per-member MLP then a set pool"))


def default_registry() -> OpRegistry:
    """A fresh registry of the built-in primitives and macros (callers may extend it)."""
    reg = OpRegistry()
    _register_primitives(reg)
    _register_macros(reg)
    return reg


def register_macro(registry: OpRegistry, name: str, args: Sequence[ArgSpec],
                   expand: Callable[["Emitter", Tuple[int, ...], Dict[str, Any]], int], *,
                   arity: int = 1, first: Callable[[Shape], bool] = lambda s: True,
                   second: Optional[Callable[[Shape, Shape], bool]] = None, doc: str = "") -> OpSpec:
    """Register a parameterized composition as a single gene.

    ``expand(emitter, sources, args)`` emits primitives (or other macros) with
    ``emitter.emit(op, *sources, **args)`` and returns the result value. Its arguments are
    enumerated like any other slot, so a macro is searched exactly like a primitive.
    """
    return registry.register(OpSpec(name, arity, tuple(args), first=first, second=second,
                                    expand=expand, doc=doc))


# ---------------------------------------------------------------------------- lowered graph
@dataclass
class Node:
    op: str                       # primitive op, or "input"
    srcs: Tuple[int, ...]
    args: Dict[str, Any]
    shape: Shape
    params: int
    row: int                      # genome row it came from (-1 for inputs)


@dataclass
class LoweredGraph:
    """Primitive DAG of a genome's effective rows plus the fixed head."""

    nodes: List[Node]
    inputs: List[int]
    output: int
    outputs: int
    head_pool: bool
    head_params: int

    @property
    def params(self) -> int:
        return sum(n.params for n in self.nodes) + self.head_params

    def describe(self) -> List[str]:
        lines = []
        for i, n in enumerate(self.nodes):
            if n.op == "input":
                lines.append(f"n{i} = input {n.shape.describe()}")
                continue
            args = ", ".join(f"{k}={v}" for k, v in n.args.items())
            srcs = ", ".join(f"n{s}" for s in n.srcs)
            lines.append(f"n{i} = {n.op}({srcs}{', ' if args else ''}{args}) -> {n.shape.describe()}")
        lines.append(f"head = {'mean-pool, ' if self.head_pool else ''}linear(n{self.output} -> {self.outputs})")
        return lines


class Emitter:
    """Appends primitives to a lowered graph while a macro expands."""

    def __init__(self, space: "ArchSpace", nodes: List[Node], row: int, depth: int = 0) -> None:
        self.space = space
        self.nodes = nodes
        self.row = row
        self.depth = depth

    def shape(self, node: int) -> Shape:
        return self.nodes[node].shape

    def emit(self, op: str, *srcs: int, **args: Any) -> int:
        spec = self.space.registry.get(op)
        if len(srcs) != spec.arity:
            raise ValueError(f"{op} takes {spec.arity} source(s)")
        shapes = tuple(self.nodes[s].shape for s in srcs)
        if not spec.first(shapes[0]) or (spec.arity == 2 and not spec.second(shapes[0], shapes[1])):
            raise ValueError(f"{op} cannot take {[s.describe() for s in shapes]}")
        missing = [a.name for a in spec.args if a.name not in args]
        if missing:
            raise ValueError(f"{op} is missing argument(s) {missing}")
        if spec.is_macro:
            if self.depth >= self.space.max_macro_nesting:
                raise ValueError(f"macro nesting deeper than {self.space.max_macro_nesting}")
            return spec.expand(Emitter(self.space, self.nodes, self.row, self.depth + 1), srcs, args)
        shape = spec.infer(shapes, args)
        self.nodes.append(Node(op, tuple(srcs), dict(args), shape, int(spec.params(shapes, args)), self.row))
        return len(self.nodes) - 1


# ---------------------------------------------------------------------------- genome
@dataclass(frozen=True)
class Row:
    op: str
    a: int
    b: int = NONE
    args: Tuple[Any, ...] = ()


@dataclass
class ArchGenome:
    rows: List[Row] = field(default_factory=list)

    def copy(self) -> "ArchGenome":
        return ArchGenome(list(self.rows))

    def to_dict(self) -> dict:
        return {"schema": "evolvo_arch_genome_v1",
                "rows": [[r.op, r.a, r.b, list(r.args)] for r in self.rows]}

    @classmethod
    def from_dict(cls, data: Mapping) -> "ArchGenome":
        if data.get("schema") != "evolvo_arch_genome_v1":
            raise ValueError("not an evolvo_arch_genome_v1 genome")
        return cls([Row(str(op), int(a), int(b), tuple(args)) for op, a, b, args in data["rows"]])


def _width_ladder(lo: int, hi: int) -> List[int]:
    """Geometric ladder with ratio sqrt(2): 4, 6, 8, 11, 16, 23, 32, 45, 64, ..."""
    values = set()
    k = 0
    while True:
        w = int(round(2 ** (k / 2)))
        if w > hi:
            break
        if w >= lo:
            values.add(w)
        k += 1
    return sorted(values)


@dataclass
class Decision:
    """One slot choice with the context the policy saw when it was (or would be) made."""

    context: Dict[str, Hashable]
    choice: Hashable


class ArchSpace:
    """The problem-specific search space: inputs, output width, budgets and option lists."""

    def __init__(
        self,
        inputs: Sequence[Tuple[str, int]],
        outputs: int,
        *,
        max_rows: int = 8,
        max_params: int = 200_000,
        width_bounds: Tuple[int, int] = (4, 1024),
        width_window: Optional[Tuple[int, int]] = None,
        max_depth: int = 6,
        max_macro_nesting: int = 4,
        ops: Optional[Sequence[str]] = None,
        activations: Sequence[str] = ("relu", "silu", "gelu", "tanh"),
        dropout: Sequence[float] = (0.0, 0.1, 0.2),
        pools: Sequence[str] = ("mean", "sum", "max"),
        ratios: Sequence[int] = (1, 2, 4),
        heads: Sequence[int] = (1, 2, 4, 8),
        registry: Optional[OpRegistry] = None,
        task: Optional[Mapping[str, Hashable]] = None,
    ) -> None:
        if not inputs:
            raise ValueError("an architecture needs at least one input")
        self.inputs = [Shape(kind, int(width), f"in{i}" if kind == SET else "")
                       for i, (kind, width) in enumerate(inputs)]
        for shape in self.inputs:
            if shape.kind not in (VEC, SET) or shape.width < 1:
                raise ValueError(f"bad input {shape}")
        if outputs < 1 or max_rows < 1 or max_params < 1 or max_depth < 1:
            raise ValueError("outputs, max_rows, max_params and max_depth must be positive")
        self.outputs = int(outputs)
        self.max_rows = int(max_rows)
        self.max_params = int(max_params)
        self.max_depth = int(max_depth)
        self.max_macro_nesting = int(max_macro_nesting)
        self.registry = registry or default_registry()
        allowed = list(ops) if ops is not None else self.registry.names()
        unknown = [name for name in allowed if name not in self.registry]
        if unknown:
            raise ValueError(f"unknown op(s) {unknown}")
        self.ops = allowed
        bad = [a for a in activations if a not in ACTIVATIONS]
        if bad or not activations:
            raise ValueError(f"activations must be a non-empty subset of {ACTIVATIONS}")
        self.lists: Dict[str, Tuple[Any, ...]] = {
            "activations": tuple(activations), "dropout": tuple(sorted(dropout)),
            "pools": tuple(pools), "ratios": tuple(sorted(ratios)), "heads": tuple(sorted(heads)),
        }
        lo, hi = int(width_bounds[0]), int(width_bounds[1])
        self.ladder = _width_ladder(max(1, lo), hi)
        if not self.ladder:
            raise ValueError("width_bounds leave no width")
        if width_window is None:
            # Start around the problem's own widths and let search widen the window.
            guess = 4 * max(max(s.width for s in self.inputs), self.outputs, 4)
            centre = min(range(len(self.ladder)), key=lambda i: abs(self.ladder[i] - guess))
            self.window = [max(0, centre - 4), min(len(self.ladder) - 1, centre + 4)]
        else:
            self.window = [self._ladder_index(width_window[0]), self._ladder_index(width_window[1])]
        self.window_events: List[dict] = []
        total_in = sum(s.width for s in self.inputs)
        self.task: Dict[str, Hashable] = {
            "t_in": log_bucket(total_in, 1), "t_out": log_bucket(self.outputs, 1),
            "t_kinds": "+".join(s.kind for s in self.inputs),
        }
        if task:
            self.task.update(task)
        self._cost_cache: Dict[tuple, Tuple[Shape, int]] = {}

    # ------------------------------------------------------------------ ladders
    def _ladder_index(self, width: int) -> int:
        return min(range(len(self.ladder)), key=lambda i: abs(self.ladder[i] - width))

    def widths(self) -> List[int]:
        return self.ladder[self.window[0]: self.window[1] + 1]

    def adapt_window(self, used: Sequence[int], *, step: int = 2) -> bool:
        """Widen the active width window where good architectures press against its edge.

        ``used`` are the width-like argument values of elite genomes. The language therefore
        grows only towards sizes the search has evidence for, and never past ``width_bounds``.
        """
        changed = False
        lo_w, hi_w = self.ladder[self.window[0]], self.ladder[self.window[1]]
        if any(w <= lo_w for w in used) and self.window[0] > 0:
            self.window[0] = max(0, self.window[0] - step)
            changed = True
        if any(w >= hi_w for w in used) and self.window[1] < len(self.ladder) - 1:
            self.window[1] = min(len(self.ladder) - 1, self.window[1] + step)
            changed = True
        if changed:
            self.window_events.append({"window": [self.ladder[self.window[0]], self.ladder[self.window[1]]]})
        return changed

    # ------------------------------------------------------------------ options
    def arg_options(self, spec: ArgSpec, first: Shape) -> List[Any]:
        if spec.domain == "width":
            return self.widths()
        if spec.domain == "depth":
            return list(range(1, self.max_depth + 1))
        if spec.domain == "choice":
            return list(spec.choices)
        if spec.domain == "divisor":
            pool = spec.choices or self.lists["heads"]
            return [h for h in pool if first.width % h == 0]
        if spec.domain.startswith("space:"):
            return list(self.lists[spec.domain[6:]])
        raise ValueError(f"unknown argument domain {spec.domain}")

    def row_cost(self, op: str, shapes: Tuple[Shape, ...], args: Dict[str, Any]) -> Tuple[Shape, int]:
        """Output shape and parameter count of one row, by lowering it on placeholders."""
        key = (op, shapes, tuple(sorted(args.items())))
        cached = self._cost_cache.get(key)
        if cached is not None:
            return cached
        nodes = [Node("input", (), {}, s, 0, -1) for s in shapes]
        out = Emitter(self, nodes, 0).emit(op, *range(len(shapes)), **args)
        result = (nodes[out].shape, sum(n.params for n in nodes))
        if len(self._cost_cache) < 100_000:
            self._cost_cache[key] = result
        return result

    def head_params(self, shape: Shape) -> int:
        return shape.width * self.outputs + self.outputs

    def _fits(self, used: int, op: str, shapes: Tuple[Shape, ...], args: Dict[str, Any]) -> bool:
        try:
            out, params = self.row_cost(op, shapes, args)
        except ValueError:
            return False
        return used + params + self.head_params(out) <= self.max_params

    def _cheapest_args(self, spec: OpSpec, first: Shape, fixed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        args = dict(fixed)
        for arg in spec.args:
            if arg.name in args:
                continue
            options = self.arg_options(arg, first)
            if not options:
                return None
            args[arg.name] = options[0]
        return args

    def source_options(self, spec: OpSpec, shapes: Sequence[Shape]) -> List[int]:
        return [i for i, s in enumerate(shapes) if spec.first(s)]

    def second_options(self, spec: OpSpec, shapes: Sequence[Shape], a: int) -> List[int]:
        return [i for i, s in enumerate(shapes) if i != a and spec.second(shapes[a], s)]

    def op_options(self, shapes: Sequence[Shape], used: int) -> List[str]:
        """Ops for which a complete row exists within the parameter budget."""
        out = []
        for name in self.ops:
            spec = self.registry.get(name)
            if self._feasible(spec, shapes, used):
                out.append(name)
        return out

    def _feasible(self, spec: OpSpec, shapes: Sequence[Shape], used: int) -> bool:
        for a in self.source_options(spec, shapes):
            bs = self.second_options(spec, shapes, a) if spec.arity == 2 else [NONE]
            for b in bs:
                src = (shapes[a],) if b == NONE else (shapes[a], shapes[b])
                args = self._cheapest_args(spec, shapes[a], {})
                if args is not None and self._fits(used, spec.name, src, args):
                    return True
        return False

    def feasible_args(self, spec: OpSpec, src: Tuple[Shape, ...], used: int,
                      fixed: Dict[str, Any], arg: ArgSpec) -> List[Any]:
        """Options of ``arg`` that still leave a within-budget row (later args cheapest)."""
        out = []
        for value in self.arg_options(arg, src[0]):
            trial = self._cheapest_args(spec, src[0], {**fixed, arg.name: value})
            if trial is not None and self._fits(used, spec.name, src, trial):
                out.append(value)
        return out

    # ------------------------------------------------------------------ decisions/contexts
    def slot_context(self, slot: str, index: int, shapes: Sequence[Shape], used: int,
                     op: Optional[str] = None, first: Optional[Shape] = None) -> Dict[str, Hashable]:
        """Features a policy may condition on. Which of them matter is learned, not declared."""
        ref = first if first is not None else shapes[-1]
        ctx: Dict[str, Hashable] = dict(self.task)
        ctx.update({
            "slot": slot,
            "kind": ref.kind,
            "w": log_bucket(ref.width, 1),
            "pos": min(3, (4 * index) // max(1, self.max_rows)),
            "room": log_bucket(max(1, self.max_params - used), 1),
        })
        if op is not None:
            ctx["op"] = op
        return ctx

    @staticmethod
    def source_choice(index_value: int, value: int, n_inputs: int) -> str:
        if value < n_inputs:
            return f"in{value}"
        distance = index_value - value
        return f"d{distance}" if distance < 4 else "d4+"

    def decisions(self, genome: ArchGenome, effective_only: bool = True) -> List[Decision]:
        """Re-derive every slot decision of a (valid) genome with its context.

        Rewards are attributed through this, so a decision is credited the same way whether
        it was sampled, inherited from a parent or produced by crossover and repair.
        """
        shapes = list(self.inputs)
        n_in = len(self.inputs)
        keep = set(self.effective_rows(genome)) if effective_only else set(range(len(genome.rows)))
        out = [Decision(self.slot_context("length", 0, shapes, 0), len(genome.rows))]
        used = 0
        for i, row in enumerate(genome.rows):
            spec = self.registry.get(row.op)
            value_index = n_in + i
            src = (shapes[row.a],) if row.b == NONE else (shapes[row.a], shapes[row.b])
            if i in keep:
                out.append(Decision(self.slot_context("op", i, shapes, used), row.op))
                ctx = self.slot_context("src_a", i, shapes, used, row.op)
                out.append(Decision(ctx, self.source_choice(value_index, row.a, n_in)))
                if row.b != NONE:
                    ctx = self.slot_context("src_b", i, shapes, used, row.op, shapes[row.a])
                    out.append(Decision(ctx, self.source_choice(value_index, row.b, n_in)))
                for arg, value in zip(spec.args, row.args):
                    ctx = self.slot_context("arg:" + arg.name, i, shapes, used, row.op, shapes[row.a])
                    out.append(Decision(ctx, value))
            shape, params = self.row_cost(row.op, src, dict(zip((a.name for a in spec.args), row.args)))
            shapes.append(shape)
            used += params
        return out

    # ------------------------------------------------------------------ sampling / repair
    def sample_row(self, index: int, shapes: Sequence[Shape], used: int, chooser: "Chooser",
                   keep: Optional[Row] = None) -> Optional[Row]:
        """Build row ``index`` slot by slot, keeping ``keep``'s choices where still valid.

        Kept numeric arguments that became invalid move to the nearest valid option (a width
        that no longer fits shrinks; heads that no longer divide the width change to the
        closest divisor) so repair preserves intent instead of re-rolling it.
        """
        op_names = self.op_options(shapes, used)
        if not op_names:
            return None
        if keep is not None and keep.op in op_names:
            op = keep.op
        else:
            keep = None
            op = chooser("op", self.slot_context("op", index, shapes, used), op_names)
        spec = self.registry.get(op)
        value_index = len(shapes)
        n_in = len(self.inputs)
        # Sources: only those that still admit a complete in-budget row.
        a_opts = [a for a in self.source_options(spec, shapes)
                  if self._source_feasible(spec, shapes, used, a)]
        if keep is not None and keep.a in a_opts:
            a = keep.a
        else:
            a = self._choose_source(chooser, "src_a", index, shapes, used, op, None, a_opts, value_index, n_in)
        b = NONE
        if spec.arity == 2:
            b_opts = [b for b in self.second_options(spec, shapes, a)
                      if self._pair_feasible(spec, shapes, used, a, b)]
            if keep is not None and keep.b in b_opts:
                b = keep.b
            else:
                b = self._choose_source(chooser, "src_b", index, shapes, used, op, shapes[a], b_opts,
                                        value_index, n_in)
        src = (shapes[a],) if b == NONE else (shapes[a], shapes[b])
        args: Dict[str, Any] = {}
        kept = dict(zip((x.name for x in spec.args), keep.args)) if keep is not None else {}
        for arg in spec.args:
            options = self.feasible_args(spec, src, used, args, arg)
            if not options:
                return None
            if arg.name in kept and kept[arg.name] in options:
                value = kept[arg.name]
            elif arg.name in kept and arg.numeric and isinstance(kept[arg.name], (int, float)):
                value = min(options, key=lambda v: (abs(math.log(max(v, 1e-9) / max(kept[arg.name], 1e-9))), v))
            else:
                ctx = self.slot_context("arg:" + arg.name, index, shapes, used, op, shapes[a])
                value = chooser("arg:" + arg.name, ctx, options)
            args[arg.name] = value
        return Row(op, a, b, tuple(args[x.name] for x in spec.args))

    def _source_feasible(self, spec: OpSpec, shapes: Sequence[Shape], used: int, a: int) -> bool:
        bs = self.second_options(spec, shapes, a) if spec.arity == 2 else [NONE]
        return any(self._pair_feasible(spec, shapes, used, a, b) for b in bs)

    def _pair_feasible(self, spec: OpSpec, shapes: Sequence[Shape], used: int, a: int, b: int) -> bool:
        src = (shapes[a],) if b == NONE else (shapes[a], shapes[b])
        args = self._cheapest_args(spec, shapes[a], {})
        return args is not None and self._fits(used, spec.name, src, args)

    def _choose_source(self, chooser, slot, index, shapes, used, op, first, options, value_index, n_in) -> int:
        labels: Dict[str, List[int]] = {}
        for opt in options:
            labels.setdefault(self.source_choice(value_index, opt, n_in), []).append(opt)
        ctx = self.slot_context(slot, index, shapes, used, op, first)
        label = chooser(slot, ctx, sorted(labels))
        group = labels[label]
        # Within "d4+" pick deterministically by the chooser's own randomness when offered.
        return group[-1] if len(group) == 1 else chooser.pick(group)

    def repair(self, genome: ArchGenome, chooser: "Chooser") -> Tuple[ArchGenome, int]:
        """Return a valid genome closest to ``genome`` and the number of rows that changed."""
        shapes = list(self.inputs)
        used = 0
        rows: List[Row] = []
        changed = 0
        for index, row in enumerate(genome.rows[: self.max_rows]):
            fixed = None
            if row.op in self.ops and 0 <= row.a < len(shapes) and (row.b == NONE or 0 <= row.b < len(shapes)):
                fixed = self.sample_row(index, shapes, used, chooser, keep=row)
            else:
                fixed = self.sample_row(index, shapes, used, chooser, keep=None)
            if fixed is None:
                changed += len(genome.rows) - index
                break
            if fixed != row:
                changed += 1
            spec = self.registry.get(fixed.op)
            src = (shapes[fixed.a],) if fixed.b == NONE else (shapes[fixed.a], shapes[fixed.b])
            shape, params = self.row_cost(fixed.op, src, dict(zip((a.name for a in spec.args), fixed.args)))
            rows.append(fixed)
            shapes.append(shape)
            used += params
        changed += max(0, len(genome.rows) - self.max_rows)
        return ArchGenome(rows), changed

    def random_genome(self, chooser: "Chooser") -> ArchGenome:
        length = chooser("length", self.slot_context("length", 0, self.inputs, 0),
                         list(range(1, self.max_rows + 1)))
        shapes = list(self.inputs)
        used = 0
        rows: List[Row] = []
        for index in range(length):
            row = self.sample_row(index, shapes, used, chooser)
            if row is None:
                break
            spec = self.registry.get(row.op)
            src = (shapes[row.a],) if row.b == NONE else (shapes[row.a], shapes[row.b])
            shape, params = self.row_cost(row.op, src, dict(zip((a.name for a in spec.args), row.args)))
            rows.append(row)
            shapes.append(shape)
            used += params
        return ArchGenome(rows)

    # ------------------------------------------------------------------ analysis
    def shapes(self, genome: ArchGenome) -> List[Shape]:
        problems = self.validate(genome)
        if problems:
            raise ValueError("; ".join(problems))
        shapes = list(self.inputs)
        for row in genome.rows:
            spec = self.registry.get(row.op)
            src = (shapes[row.a],) if row.b == NONE else (shapes[row.a], shapes[row.b])
            shapes.append(self.row_cost(row.op, src, dict(zip((a.name for a in spec.args), row.args)))[0])
        return shapes

    def validate(self, genome: ArchGenome) -> List[str]:
        """Every reason ``genome`` is not a member of this language (empty when valid)."""
        problems: List[str] = []
        shapes = list(self.inputs)
        used = 0
        if len(genome.rows) > self.max_rows:
            problems.append(f"{len(genome.rows)} rows exceed max_rows {self.max_rows}")
        for i, row in enumerate(genome.rows):
            where = f"row {i} ({row.op})"
            if row.op not in self.ops:
                problems.append(f"{where}: op not in this space")
                break
            spec = self.registry.get(row.op)
            if not 0 <= row.a < len(shapes) or not spec.first(shapes[row.a]):
                problems.append(f"{where}: source a={row.a} is not valid")
                break
            if spec.arity == 2:
                if not (0 <= row.b < len(shapes)) or row.b == row.a or not spec.second(shapes[row.a], shapes[row.b]):
                    problems.append(f"{where}: source b={row.b} is not valid")
                    break
            elif row.b != NONE:
                problems.append(f"{where}: unary op with a second source")
                break
            if len(row.args) != len(spec.args):
                problems.append(f"{where}: expected {len(spec.args)} argument(s)")
                break
            src = (shapes[row.a],) if row.b == NONE else (shapes[row.a], shapes[row.b])
            for arg, value in zip(spec.args, row.args):
                if value not in self.arg_options(arg, shapes[row.a]):
                    problems.append(f"{where}: {arg.name}={value} is not an enumerated option")
            if problems:
                break
            args = dict(zip((a.name for a in spec.args), row.args))
            shape, params = self.row_cost(row.op, src, args)
            used += params
            shapes.append(shape)
        if not problems:
            total = self.lower(genome).params
            if total > self.max_params:
                problems.append(f"{total} parameters exceed max_params {self.max_params}")
        return problems

    def output_value(self, genome: ArchGenome) -> int:
        return len(self.inputs) + len(genome.rows) - 1 if genome.rows else 0

    def effective_rows(self, genome: ArchGenome) -> List[int]:
        """Rows that reach the output (ascending); the rest are introns."""
        n_in = len(self.inputs)
        if not genome.rows:
            return []
        live = {self.output_value(genome)}
        rows = []
        for i in range(len(genome.rows) - 1, -1, -1):
            if n_in + i in live:
                rows.append(i)
                row = genome.rows[i]
                live.add(row.a)
                if row.b != NONE:
                    live.add(row.b)
        return sorted(rows)

    def lower(self, genome: ArchGenome) -> LoweredGraph:
        """Expand the effective rows (macros included) to a primitive graph plus head."""
        nodes = [Node("input", (), {}, s, 0, -1) for s in self.inputs]
        value_node: Dict[int, int] = {i: i for i in range(len(self.inputs))}
        n_in = len(self.inputs)
        for i in self.effective_rows(genome):
            row = genome.rows[i]
            spec = self.registry.get(row.op)
            srcs = (value_node[row.a],) if row.b == NONE else (value_node[row.a], value_node[row.b])
            out = Emitter(self, nodes, i).emit(row.op, *srcs, **dict(zip((a.name for a in spec.args), row.args)))
            value_node[n_in + i] = out
        output = value_node[self.output_value(genome)]
        shape = nodes[output].shape
        return LoweredGraph(nodes, list(range(n_in)), output, self.outputs,
                            head_pool=shape.kind == SET, head_params=self.head_params(shape))

    def signature(self, genome: ArchGenome) -> str:
        """Canonical text of the effective architecture (cache key; introns excluded)."""
        n_in = len(self.inputs)
        eff = self.effective_rows(genome)
        renumber = {i: i for i in range(n_in)}
        parts = []
        for k, i in enumerate(eff):
            row = genome.rows[i]
            renumber[n_in + i] = n_in + k
            srcs = ",".join(f"v{renumber[s]}" for s in ((row.a,) if row.b == NONE else (row.a, row.b)))
            spec = self.registry.get(row.op)
            args = ",".join(f"{a.name}={v}" for a, v in zip(spec.args, row.args))
            parts.append(f"{row.op}({srcs}{';' if args else ''}{args})")
        return "|".join(parts) or "identity"

    def to_slots(self, genome: ArchGenome) -> List[Tuple[int, ...]]:
        """The fixed 7-slot integer encoding: target, op code, a, b, arg option indices."""
        shapes = self.shapes(genome)
        out = []
        n_in = len(self.inputs)
        for i, row in enumerate(genome.rows):
            spec = self.registry.get(row.op)
            idx = [self.arg_options(arg, shapes[row.a]).index(v) for arg, v in zip(spec.args, row.args)]
            idx += [NONE] * (MAX_ARGS - len(idx))
            out.append((n_in + i, self.registry.code(row.op), row.a, row.b, *idx))
        return out

    def describe(self, genome: ArchGenome) -> List[str]:
        n_in = len(self.inputs)
        eff = set(self.effective_rows(genome))
        shapes = self.shapes(genome)
        lines = [f"v{i} = input {s.describe()}" for i, s in enumerate(self.inputs)]
        for i, row in enumerate(genome.rows):
            spec = self.registry.get(row.op)
            srcs = ", ".join(f"v{s}" for s in ((row.a,) if row.b == NONE else (row.a, row.b)))
            args = ", ".join(f"{a.name}={v}" for a, v in zip(spec.args, row.args))
            mark = "" if i in eff else "   # intron"
            lines.append(f"v{n_in + i} = {row.op}({srcs}{', ' if args else ''}{args}) -> "
                         f"{shapes[n_in + i].describe()}{mark}")
        out = shapes[self.output_value(genome)]
        lines.append(f"out = {'mean-pool, ' if out.kind == SET else ''}linear(v{self.output_value(genome)} -> {self.outputs})")
        return lines

    def width_values(self, genome: ArchGenome) -> List[int]:
        """Width-ladder argument values used by effective rows (for window adaptation)."""
        out = []
        for i in self.effective_rows(genome):
            row = genome.rows[i]
            for arg, value in zip(self.registry.get(row.op).args, row.args):
                if arg.domain == "width":
                    out.append(int(value))
        return out


class Chooser:
    """Slot-choice callback: ``chooser(slot, context, options) -> option``.

    The default chooses uniformly with its own ``random.Random``; ``evolvo.nnarch.policy``
    provides the learned one.
    """

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng or random.Random(0)

    def __call__(self, slot: str, context: Mapping[str, Hashable], options: Sequence[Any]) -> Any:
        if not options:
            raise ValueError(f"no options for {slot}")
        return options[self.rng.randrange(len(options))]

    def pick(self, options: Sequence[Any]) -> Any:
        return options[self.rng.randrange(len(options))]


__all__ = [
    "ACTIVATIONS", "ArchGenome", "ArchSpace", "ArgSpec", "Chooser", "Decision", "Emitter",
    "LoweredGraph", "Node", "NONE", "OpRegistry", "OpSpec", "Row", "SET", "SLOT_COUNT", "Shape",
    "VEC", "default_registry", "register_macro",
]
