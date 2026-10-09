import numpy as np
from .arm import fk, step, sample_target, sample_start, MAX_STEPS, REACH_TOL
from .expert import expert_action
from .policy import obs


def rollout(act_fn, q0, target, max_steps=MAX_STEPS):
    """Run act_fn(q, target) -> dq. Returns joint path and final distance to target."""
    q, path = q0.copy(), [q0.copy()]
    for _ in range(max_steps):
        if np.linalg.norm(target - fk(q)) < REACH_TOL:
            break
        q = step(q, act_fn(q, target))
        path.append(q.copy())
    return np.array(path), float(np.linalg.norm(target - fk(q)))


def collect_expert(n_demos, rng, noise=0.01):
    X, Y = [], []
    for _ in range(n_demos):
        q0, tgt = sample_start(rng), sample_target(rng)
        path, _ = rollout(lambda q, t: expert_action(q, t, rng, noise), q0, tgt)
        for q in path[:-1]:
            X.append(obs(q, tgt)); Y.append(expert_action(q, tgt))   # label = clean expert action
    return np.array(X), np.array(Y, dtype=np.float32)


def collect_dagger(policy, n_demos, rng):
    """Roll out the LEARNER, but label every visited state with the expert's action."""
    X, Y = [], []
    for _ in range(n_demos):
        q0, tgt = sample_start(rng), sample_target(rng)
        path, _ = rollout(policy.act, q0, tgt)
        for q in path[:-1]:
            X.append(obs(q, tgt)); Y.append(expert_action(q, tgt))
    return np.array(X), np.array(Y, dtype=np.float32)


def evaluate(act_fn, n=500, seed=999):
    """Success rate and mean final error on held-out (start, target) pairs."""
    rng = np.random.default_rng(seed)
    errs = [rollout(act_fn, sample_start(rng), sample_target(rng))[1] for _ in range(n)]
    errs = np.array(errs)
    return {"success_rate": float((errs < REACH_TOL).mean()), "mean_final_error_m": float(errs.mean())}
