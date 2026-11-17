"""GFSL-NN architecture search on a small synthetic regression (needs PyTorch).

    python examples/arch_search_flow.py [knowledge.json]

Evolves torch architectures (MLP/residual/transformer macros and primitives), trains each with
learned plateau stopping, and prints the best genome, the size/loss Pareto front and what the
search learned. Pass a knowledge file to start from (and update) earlier searches' tendencies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from evolvo.arch import ArchSpace, ArchitectureSearch, Knowledge  # noqa: E402
from evolvo.arch.train import TorchTask  # noqa: E402


def main() -> None:
    knowledge_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    knowledge = Knowledge.load(knowledge_path) if knowledge_path and knowledge_path.exists() else None
    torch.manual_seed(0)
    x = torch.rand(700, 4) * 2 - 1
    y = torch.sin(3 * x[:, :1]) * x[:, 1:2] + x[:, 2:3] ** 2 - 0.5 * x[:, 3:4]
    task = TorchTask((x[:400], y[:400]), (x[400:550], y[400:550]), lr=5e-3, hard_max_epochs=600)
    space = ArchSpace([("vec", 4)], 1, max_rows=5, max_params=30_000)
    search = ArchitectureSearch(space, task, seed=1, population=8, knowledge=knowledge)
    report = search.run(generations=10, patience=4,
                        progress=lambda g: print(f"gen {g['generation']}: best loss {g['best_loss']:.5f} "
                                                 f"({g['best_params']} params), widths {g['window']}"))
    print(f"\nstopped: {report['stop_reason']} after {report['evaluations']} evaluations")
    print("\n".join(report["best"]["describe"]))
    model = task.best_module
    with torch.no_grad():
        test = float(torch.nn.functional.mse_loss(model(x[550:]), y[550:]))
    print(f"held-out MSE {test:.5f} (target variance {float(y[550:].var()):.5f})")
    print("pareto:", [(p["params"], round(p["loss"], 5)) for p in report["pareto"]])
    print("policy feature order:", report["policy"]["stats"]["order"])
    print("epoch-stat feature order:", report["epochs"]["stats"]["order"])
    if knowledge_path:
        search.knowledge.save(knowledge_path)
        print("knowledge saved to", knowledge_path)


if __name__ == "__main__":
    main()
