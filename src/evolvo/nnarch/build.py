"""Lower a GFSL-NN genome to a ``torch.nn.Module`` (requires PyTorch).

The module is a straight-line program over the lowered primitive graph: one entry per node,
evaluated in order, plus the fixed head. It is deterministic given ``torch.manual_seed`` and is
traceable with ``torch.jit.trace`` (no data-dependent control flow), so it can be exported as
TorchScript.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .language import ArchGenome, ArchSpace, LoweredGraph

try:  # pragma: no cover - exercised only with torch present
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _IMPORT_ERROR: Optional[ImportError] = exc
else:
    _IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("evolvo.nnarch.build needs PyTorch (`pip install torch`)") from _IMPORT_ERROR


def _activation(name: str):
    return {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid}[name]()


if nn is not None:

    class _Tokenize(nn.Module):
        """Per-feature embedding: ``[B, F] -> [B, F, dim]`` with ``x_f * W_f + b_f``."""

        def __init__(self, features: int, dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(features, dim) * (1.0 / dim) ** 0.5)
            self.bias = nn.Parameter(torch.zeros(features, dim))

        def forward(self, x):
            return x.unsqueeze(-1) * self.weight + self.bias

    class _Attend(nn.Module):
        def __init__(self, width: int, heads: int) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(width, heads, batch_first=True)

        def forward(self, x):
            return self.attn(x, x, x, need_weights=False)[0]

    class GenomeModule(nn.Module):
        """Executes a :class:`LoweredGraph`; ``forward(*inputs)`` takes one tensor per input."""

        def __init__(self, graph: LoweredGraph) -> None:
            super().__init__()
            self.graph = graph
            self.steps: List[tuple] = []
            self.layers = nn.ModuleDict()
            for i, node in enumerate(graph.nodes):
                if node.op == "input":
                    continue
                src_shape = graph.nodes[node.srcs[0]].shape
                module = None
                if node.op == "linear":
                    module = nn.Linear(src_shape.width, node.args["units"])
                elif node.op == "act":
                    module = _activation(node.args["fn"])
                elif node.op == "norm":
                    module = nn.LayerNorm(src_shape.width)
                elif node.op == "dropout":
                    module = nn.Dropout(float(node.args["rate"]))
                elif node.op == "attend":
                    module = _Attend(src_shape.width, int(node.args["heads"]))
                elif node.op == "tokenize":
                    module = _Tokenize(src_shape.width, node.args["dim"])
                elif node.op not in ("add", "mul", "concat", "pool", "context"):
                    raise ValueError(f"no torch lowering for primitive {node.op!r}")
                if module is not None:
                    self.layers[f"n{i}"] = module
                self.steps.append((i, node.op, node.srcs, node.args.get("mode")))
            out = graph.nodes[graph.output].shape
            self.head = nn.Linear(out.width, graph.outputs)
            self.n_inputs = len(graph.inputs)

        def forward(self, *inputs):
            values = {i: x for i, x in enumerate(inputs)}
            for i, op, srcs, mode in self.steps:
                a = values[srcs[0]]
                if op == "add":
                    y = a + values[srcs[1]]
                elif op == "mul":
                    y = a * values[srcs[1]]
                elif op == "concat":
                    y = torch.cat([a, values[srcs[1]]], dim=-1)
                elif op == "pool":
                    if mode == "sum":
                        y = a.sum(dim=1)
                    elif mode == "max":
                        y = a.max(dim=1).values
                    else:
                        y = a.mean(dim=1)
                elif op == "context":
                    b = values[srcs[1]]
                    y = torch.cat([a, b.unsqueeze(1).expand(-1, a.shape[1], -1)], dim=-1)
                else:
                    y = self.layers[f"n{i}"](a)
                values[i] = y
            out = values[self.graph.output]
            if self.graph.head_pool:
                out = out.mean(dim=1)
            return self.head(out)

else:  # pragma: no cover
    GenomeModule = None  # type: ignore[assignment]


def build_module(space: ArchSpace, genome: ArchGenome, *, seed: Optional[int] = None):
    """Build the torch module of ``genome`` (initialization seeded when ``seed`` is given)."""
    _require_torch()
    graph = space.lower(genome)
    if seed is not None:
        torch.manual_seed(int(seed))
    return GenomeModule(graph)


def parameter_count(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def trace_module(module, example_inputs: Sequence):
    """TorchScript trace in eval mode (dropout off); callers verify parity on a new batch size."""
    _require_torch()
    module.eval()
    with torch.no_grad():
        return torch.jit.trace(module, tuple(example_inputs), check_trace=False)


__all__ = ["GenomeModule", "build_module", "parameter_count", "trace_module"]
