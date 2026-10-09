"""Planar 3-joint arm: kinematics, Jacobian, and a reach task.

Modeled on my potentiometer-driven servo arm (base/shoulder/wrist). The sim is
planar to keep the math small. The shoulder has a floor limit like the real arm.
"""
import numpy as np

LINKS = np.array([0.30, 0.25, 0.15])             # metres (sim parameters)
Q_MIN = np.array([0.0, -2.4, -2.0])              # rad; shoulder floor = 0 (no digging into the table)
Q_MAX = np.array([np.pi, 2.4, 2.0])
MAX_STEP = 0.05                                  # rad per control step
REACH_TOL = 0.02                                 # m, counts as success
MAX_STEPS = 100


def fk(q):
    """End-effector (x, y) for joint angles q."""
    a = np.cumsum(q)
    return np.array([np.sum(LINKS * np.cos(a)), np.sum(LINKS * np.sin(a))])


def jacobian(q):
    """2x3 Jacobian d(x,y)/dq."""
    a = np.cumsum(q)
    J = np.zeros((2, 3))
    for j in range(3):
        J[0, j] = -np.sum(LINKS[j:] * np.sin(a[j:]))
        J[1, j] = np.sum(LINKS[j:] * np.cos(a[j:]))
    return J


def sample_target(rng):
    """A target that is reachable by construction: the end-effector position of a
    random valid pose, kept above the table (y > 0.05)."""
    while True:
        p = fk(rng.uniform(Q_MIN, Q_MAX))
        if p[1] > 0.05:
            return p


def sample_start(rng):
    return rng.uniform(Q_MIN + 0.2, Q_MAX - 0.2)


def step(q, dq):
    """Apply a joint-velocity command with per-step and joint limits."""
    dq = np.clip(dq, -MAX_STEP, MAX_STEP)
    return np.clip(q + dq, Q_MIN, Q_MAX)
