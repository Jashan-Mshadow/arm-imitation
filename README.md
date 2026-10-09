# arm-imitation

Imitation learning for a simulated 3-joint robot arm. A scripted "teleoperator" records
demonstrations, a PyTorch policy learns from them, and DAgger fixes the mistakes that
plain imitation makes. Small on purpose: every part fits in your head.

The arm is modeled on my potentiometer-driven servo arm (`popsicle-robot-arm`), as a
planar 3-link sim with the same idea of joint limits (the shoulder has a floor so it
cannot dig into the table).

## The idea

Robot teams like Tesla's Optimus collect human teleoperation demonstrations and train
policies from them. This repo is that loop in miniature:

1. **Sim** (`arm.py`): forward kinematics, Jacobian, joint and step limits, a reach task.
2. **Teleoperator** (`expert.py`): damped least-squares inverse kinematics that freezes a
   joint sitting on a limit and re-solves with the others. Stands in for a human.
3. **Behavior cloning** (`policy.py`): an MLP maps (3 joint angles, 2D target) to 3 joint
   velocity commands, trained with MSE on the demonstrations.
4. **DAgger** (`rollout.py`, `run.py`): let the *learner* drive, have the expert relabel the
   states it visits, add them to the data, retrain. Five rounds.

## Results

500 held-out reaches (random start pose + reachable target, success = end effector within
2 cm in 100 steps). Seed 0, `results/results.json`, plot in `results/curves.png`.

| Policy | Success | Mean final error |
|---|---|---|
| Scripted expert (the ceiling) | 93.2% | 1.9 cm |
| Behavior cloning, 100 demos (3,652 steps) | 23.8% | 13.6 cm |
| DAgger, 5 rounds (15,759 steps) | 41.8% | 6.2 cm |

**What this shows.** Plain behavior cloning is bad even though the training loss is low:
once the policy makes a small error it lands in states the demos never covered, and errors
compound (covariate shift). Letting the learner visit its own states and getting expert
labels there (DAgger) raises success from 24% to 42% and cuts mean final error by 54%.

**Honest limits.**
- It is still far below the expert (42% vs 93%). The policy is a small MLP on 5 inputs
  and I did not tune it much (more epochs helped, an extra error-feature input did not).
  Those two choices were made on a separate seed, not the 500-reach test seed.
- The "human" is a script, not a person, and the sim is planar with no physics or contact.
- Single seed. I have not measured variance.
- The expert itself fails on 7% of reaches (it can stall near joint limits).

## Run it

    uv venv .venv && uv pip install --python .venv/bin/python numpy torch matplotlib pytest
    PYTHONPATH=src .venv/bin/python -m arm_imitation.run     # about 40 s on a laptop CPU
    .venv/bin/python -m pytest -q                            # 5 tests

## Next

- Add contact and a gripper in MuJoCo, then a pick-and-place task.
- Record real demonstrations from the pot-controlled arm instead of a scripted expert.
- Try a larger policy / more data to close the gap to the expert; report seeds and variance.
