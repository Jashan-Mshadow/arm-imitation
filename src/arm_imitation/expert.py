"""The scripted 'teleoperator': damped least-squares IK toward the target.

A joint sitting on a limit and being pushed further out is frozen and the step
is re-solved with the remaining joints (active-set), so the expert does not get
pinned against a limit.
"""
import numpy as np
from .arm import fk, jacobian, MAX_STEP, Q_MIN, Q_MAX

DAMPING = 0.05


def _solve(J, err):
    return J.T @ np.linalg.solve(J @ J.T + DAMPING**2 * np.eye(2), err)


def expert_action(q, target, rng=None, noise=0.0):
    err = target - fk(q)
    J = jacobian(q)
    dq = _solve(J, err)
    pinned = ((q <= Q_MIN + 1e-9) & (dq < 0)) | ((q >= Q_MAX - 1e-9) & (dq > 0))
    if pinned.any():
        Jf = J.copy(); Jf[:, pinned] = 0.0
        dq = _solve(Jf, err); dq[pinned] = 0.0
    dq = np.clip(dq, -MAX_STEP, MAX_STEP)
    if rng is not None and noise > 0:
        dq = dq + rng.normal(0, noise, size=3)   # imperfect human hand
    return np.clip(dq, -MAX_STEP, MAX_STEP)
