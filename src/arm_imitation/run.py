"""Train behavior cloning, then DAgger, and write results/results.json + results/curves.png."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .arm import sample_target, sample_start
from .expert import expert_action
from .policy import Policy, train
from .rollout import collect_expert, collect_dagger, evaluate, rollout

SEED, N_DEMOS, DAGGER_ROUNDS = 0, 100, 5


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    expert_fn = lambda q, t: expert_action(q, t)
    res = {"expert": evaluate(expert_fn)}

    X, Y = collect_expert(N_DEMOS, rng)
    res["demo_samples"] = len(X)
    bc = Policy()
    train(bc, X, Y)
    res["behavior_cloning"] = evaluate(bc.act)

    # DAgger: add the learner's own visited states, labeled by the expert, and retrain.
    pol, Xd, Yd, curve = Policy(), X.copy(), Y.copy(), []
    for r in range(DAGGER_ROUNDS):
        train(pol, Xd, Yd, epochs=300 if r == 0 else 150)   # later rounds warm-start
        curve.append(evaluate(pol.act)["success_rate"])
        Xn, Yn = collect_dagger(pol, 30, rng)
        Xd, Yd = np.concatenate([Xd, Xn]), np.concatenate([Yd, Yn])
    train(pol, Xd, Yd)
    res["dagger"] = evaluate(pol.act)
    res["dagger"]["samples"] = len(Xd)
    res["dagger_success_by_round"] = curve + [res["dagger"]["success_rate"]]

    json.dump(res, open("results/results.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    names = ["expert", "behavior_cloning", "dagger"]
    ax[0].bar(["expert", "BC", "DAgger"], [res[n]["success_rate"] * 100 for n in names])
    ax[0].set_ylabel("success rate (%)"); ax[0].set_title("500 held-out reaches")
    ax[1].plot(range(len(res["dagger_success_by_round"])), np.array(res["dagger_success_by_round"]) * 100, marker="o")
    ax[1].axhline(res["behavior_cloning"]["success_rate"] * 100, ls="--", c="gray", label="BC only")
    ax[1].set_xlabel("DAgger round"); ax[1].set_ylabel("success rate (%)"); ax[1].legend()
    fig.tight_layout(); fig.savefig("results/curves.png", dpi=130)


if __name__ == "__main__":
    main()
