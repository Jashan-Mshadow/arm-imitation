import numpy as np
from arm_imitation.arm import fk, jacobian, step, sample_target, sample_start, Q_MIN, Q_MAX, MAX_STEP
from arm_imitation.expert import expert_action
from arm_imitation.rollout import rollout


def test_fk_straight_arm():
    assert np.allclose(fk(np.zeros(3)), [0.70, 0.0])


def test_jacobian_matches_finite_difference():
    q = np.array([0.7, -0.4, 0.3]); eps = 1e-6
    J = jacobian(q)
    for j in range(3):
        d = np.zeros(3); d[j] = eps
        assert np.allclose((fk(q + d) - fk(q - d)) / (2 * eps), J[:, j], atol=1e-5)


def test_step_respects_limits():
    q = step(Q_MAX.copy(), np.ones(3))
    assert np.all(q <= Q_MAX) and np.all(q >= Q_MIN)
    assert np.all(np.abs(step(np.zeros(3) + 1.0, np.ones(3) * 5) - 1.0) <= MAX_STEP + 1e-12)


def test_targets_are_above_table():
    rng = np.random.default_rng(1)
    assert all(sample_target(rng)[1] > 0.05 for _ in range(200))


def test_expert_reaches_most_targets():
    rng = np.random.default_rng(2)
    ok = sum(rollout(lambda q, t: expert_action(q, t), sample_start(rng), sample_target(rng))[1] < 0.02 for _ in range(100))
    assert ok >= 85
