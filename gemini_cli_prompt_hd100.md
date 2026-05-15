You are an uncompromising MARL expert and senior Python engineer. I am a student at Monash University working on FIT5226 Stage 2. My current submission scores approximately 70/100 — Distinction level. I need a complete, HD-quality (100/100) rewrite of my simulation as a **single, fully-executed Jupyter Notebook** that can be submitted directly.

The notebook filename must be: `FIT5226_34169202_2026S1_Ass2.ipynb`

## MANDATORY CONSTRAINTS — DO NOT DEVIATE

1. **Single standalone notebook.** No external `.py` file imports at runtime. Every class and function must be defined inline in the notebook cells. The existing `.py` modules in `/src` are reference implementations only.
2. **The notebook MUST be fully executed before saving.** Every code cell must have non-empty `outputs` and a sequential `execution_count`. Learning curves, policy plots, EGT plots, and printed metrics must all be visible as cell outputs. A notebook with empty outputs will receive zero credit.
3. **Tabular Q-Learning only.** No DQN, no neural networks, no function approximation.
4. **No reward heuristics.** No distance-based shaping, no shaped potentials. Only the exact reward values specified.
5. **AI Declaration cell must be present** and must accurately describe what AI was used for. Do not remove it.

---

## EXACT BUG FIXES REQUIRED IN THE REWRITE

### BUG 1 — Missing `wait_cost` in Phase 2 config (CRASH BUG)
The `EnvConfig` dataclass requires `wait_cost` as a non-optional field. The Phase 2 config dict `STAGE2_PHASE2_ROBERT_CONFIG` must include `"wait_cost": -3.0`. Without this, Phase 2 instantiation throws `TypeError` and the entire Task 2 section produces no output.

### BUG 2 — Collision fires before delivery (REWARD CORRUPTION BUG)
In `StochasticMultiAgentEnv.step()`, the collision check currently runs before the per-agent task-completion loop. If both agents simultaneously step onto their respective delivery cells AND those cells happen to be the same coordinate (impossible in this layout, but the logic is still wrong in principle), or if an agent is marked `done` inside the task loop but the collision was already counted, the penalty is applied incorrectly. **Fix: move the collision check to AFTER the task-completion loop, and guard with `if not self.done[agent_id]` for both agents.**

### BUG 3 — `p_hat_flip` is shared state and biased early (CONVERGENCE BUG)
The current `TabularQAgent` computes `p_hat_flip` by counting lake-state changes across the entire episode. In the first ~50 episodes, `p_hat_flip` is far from the true `p=0.5`, making the expected-value update inaccurate and slowing convergence. **Fix: pass `p_flood` from `EnvConfig` directly into `TabularQAgent.__init__` as a known constant and use it directly in the Bellman update instead of the empirically estimated `p_hat_flip`.** The assignment does NOT say the agent must estimate p — it says to "account for both possible states." Using the known p is correct and faster.

### BUG 4 — Robert config missing `wait_cost` AND `pickup_reward` fields (CRASH BUG)
Same as Bug 1 — ensure `STAGE2_PHASE2_ROBERT_CONFIG` includes both `"wait_cost": -3.0` and `"pickup_reward": 10.0`.

### BUG 5 — Evaluation sample size too small (METRIC BUG)
`evaluate_greedy` uses `episodes=20`. For a stochastic environment with p=0.5 lake flipping, 20 episodes produces high-variance estimates. **Change to `episodes=100` for all evaluations.** This is necessary for the convergence plots to show smooth curves that satisfy "convincing evidence" for Credit/HD.

---

## COMPLETE NOTEBOOK STRUCTURE REQUIRED

### Cell 1 — Markdown: Title and student metadata
```
# FIT5226 Project Stage 2: Multi-Agent Stochastic Games
**Student Name:** Juan Pulido
**Student ID:** 34169202
```

### Cell 2 — Markdown: Table of Contents
Include sections: AI Declaration, Architecture, Mathematical Formulation, Implementation, Task 1 Training, Task 2 Training, HD Experiment, EGT Model, Task 4 Theory.

### Cell 3 — Markdown: AI Declaration
Accurately state that Gemini CLI was used as an architectural co-pilot and auditor. Confirm the student understands and can defend every line.

### Cell 4 — Markdown: Architecture and Mathematical Formulation
Include:
- MMDP formulation: state space S, action space A, transition dynamics P, reward function R
- The Expected-Value Bellman equation written in LaTeX: `$Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \left((1-p)\max_{a'}Q(s'_{dry}, a') + p\max_{a'}Q(s'_{flooded}, a')\right) - Q(s,a)\right]$`
- Explanation of WHY this update is needed (lake can flip between t and t+1)
- Partial observability justification: state = (row, col, has_sample, lake_flooded), no opponent location
- Argument for why independent learning is valid here (hint 5 compliance): agents have non-overlapping primary routes; the shared lake is the only interaction point; the lake state is observable; therefore each agent can learn a best-response conditional on lake state independently, and the equilibrium reached by sequential independent learners is the same as the joint equilibrium because Phase 1 has a unique dominant strategy profile

### Cell 5 — Code: Imports and seed
```python
import numpy as np, random, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Tuple, List, Dict, Any
from scipy.integrate import odeint
np.random.seed(42); random.seed(42)
```

### Cell 6 — Code: Config dataclasses
Define `AgentConfig`, `EnvConfig`, `ExperimentConfig` exactly as in the existing codebase. Add `from_dict` classmethod to `ExperimentConfig`. All fields required in `EnvConfig`: `grid_rows`, `grid_cols`, `p_flood`, `step_cost`, `wait_cost`, `pickup_reward`, `success_reward`, `collision_penalty`, `hazard_penalty`.

Then define the three inline config dicts:

**PHASE 1 CONFIG** (`STAGE2_PHASE1_CONFIG`):
```python
{
    "experiment_name": "Phase1_Sarah_Safe",
    "is_multi_agent": True,
    "training_episode_budget": 8000,
    "agent": {
        "learning_rate_alpha": 0.1,
        "discount_factor_gamma": 0.95,
        "initial_epsilon": 1.0,
        "epsilon_decay_rate": 0.9995,
        "minimum_epsilon": 0.01,
        "action_size": 5
    },
    "env": {
        "grid_rows": 5, "grid_cols": 5, "p_flood": 0.5,
        "step_cost": -5.0, "wait_cost": -3.0,
        "pickup_reward": 10.0, "success_reward": 50.0,
        "collision_penalty": -20.0, "hazard_penalty": -20.0
    }
}
```

**PHASE 2 CONFIG** (`STAGE2_PHASE2_CONFIG`):
```python
{
    "experiment_name": "Phase2_Robert_Symmetric",
    "is_multi_agent": True,
    "training_episode_budget": 15000,
    "agent": {
        "learning_rate_alpha": 0.1,
        "discount_factor_gamma": 0.95,
        "initial_epsilon": 1.0,
        "epsilon_decay_rate": 0.9997,
        "minimum_epsilon": 0.05,
        "action_size": 5
    },
    "env": {
        "grid_rows": 5, "grid_cols": 5, "p_flood": 0.5,
        "step_cost": -5.0, "wait_cost": -3.0,
        "pickup_reward": 10.0, "success_reward": 50.0,
        "collision_penalty": -20.0, "hazard_penalty": 0.0
    }
}
```

### Cell 7 — Code: Actions and State Handler
Copy `Directions(IntEnum)`, `Actions` class, and `StateHandler` from existing codebase exactly. No changes needed.

### Cell 8 — Code: Environment (`StochasticMultiAgentEnv`)
Rewrite `step()` with the following corrected ordering:

```
def step(self, joint_action):
    rewards = {aid: 0.0 for aid in self.agent_types}

    # 1. Lake stochastic transition (BEFORE movement)
    if random.random() < self.config.p_flood:
        self.lake_flooded = not self.lake_flooded

    # 2. Compute intended next positions (simultaneous, based on PREVIOUS positions)
    prev_positions = self.positions.copy()
    new_positions = {}
    for agent_id, action in joint_action.items():
        if self.done[agent_id]:
            new_positions[agent_id] = prev_positions[agent_id]
            continue
        candidate = Actions.apply_action(prev_positions[agent_id], action)
        if Actions.is_valid_move(candidate, self.grid_size):
            new_positions[agent_id] = candidate
        else:
            new_positions[agent_id] = prev_positions[agent_id]
        rewards[agent_id] = self.config.wait_cost if action == Directions.WAIT else self.config.step_cost

    self.positions = new_positions

    # 3. Task completion (BEFORE collision check — done status must be set first)
    for agent_id in ["Agent_A", "Agent_B"]:
        if self.done[agent_id]: continue
        pos = self.positions[agent_id]
        atype = self.agent_types[agent_id]
        if atype == "Type_A":
            if not self.has_sample[agent_id] and pos == self.locs["U"]:
                self.has_sample[agent_id] = True
                rewards[agent_id] += self.config.pickup_reward
            elif self.has_sample[agent_id] and pos == self.locs["X"]:
                self.done[agent_id] = True
                rewards[agent_id] += self.config.success_reward
        else:
            if not self.has_sample[agent_id] and pos == self.locs["V"]:
                self.has_sample[agent_id] = True
                rewards[agent_id] += self.config.pickup_reward
            elif self.has_sample[agent_id] and pos == self.locs["Y"]:
                self.done[agent_id] = True
                rewards[agent_id] += self.config.success_reward

    # 4. Collision check (AFTER task completion, only on agents still active)
    if not self.done["Agent_A"] and not self.done["Agent_B"]:
        if self.positions["Agent_A"] == self.positions["Agent_B"]:
            rewards["Agent_A"] += self.config.collision_penalty
            rewards["Agent_B"] += self.config.collision_penalty

    # 5. Water hazard (Agent A only, only if still active)
    if not self.done["Agent_A"] and self.lake_flooded and self.positions["Agent_A"] == self.locs["Lake"]:
        rewards["Agent_A"] += self.config.hazard_penalty

    all_done = all(self.done.values())
    return self._get_obs(), rewards, self.done.copy(), all_done
```

### Cell 9 — Code: `TabularQAgent`
Use the fixed version with `p_flood` passed as a constructor parameter instead of `p_hat_flip`:

```python
class TabularQAgent:
    def __init__(self, agent_id, config, p_flood):
        self.agent_id = agent_id
        self.config = config
        self.p_flood = p_flood  # Known environment parameter — use directly
        self.q_table = defaultdict(lambda: np.zeros(config.action_size))
        self.epsilon = config.initial_epsilon

    def update_learning(self, state, action, reward, next_obs_state, terminal):
        current_lake = state[3]
        y_next, x_next, payload_next, _ = next_obs_state
        current_q = self.q_table[state][action]
        if terminal:
            expected_future_q = 0.0
        else:
            p = self.p_flood
            state_stays = (y_next, x_next, payload_next, current_lake)
            state_flips = (y_next, x_next, payload_next, not current_lake)
            max_q_stays = np.max(self.q_table[state_stays])
            max_q_flips = np.max(self.q_table[state_flips])
            expected_future_q = (1 - p) * max_q_stays + p * max_q_flips
        td_target = reward + self.config.discount_factor_gamma * expected_future_q
        self.q_table[state][action] += self.config.learning_rate_alpha * (td_target - current_q)

    def choose_action(self, observation, greedy=False):
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.config.action_size))
        return int(np.argmax(self.q_table[observation]))

    def decay_epsilon(self):
        self.epsilon = max(self.config.minimum_epsilon, self.epsilon * self.config.epsilon_decay_rate)
```

### Cell 10 — Code: `SimulationRunner` and `evaluate_greedy`
Keep the existing runner logic but change `evaluate_greedy` to use `episodes=100` and return per-step rates (divide by `steps * episodes`, not just `episodes`). Also add `avg_steps_to_complete` metric.

### Cell 11 — Code: Policy visualiser (`visualize_policy`)
Keep the existing quiver-plot implementation. Add a second visualisation function `compare_policies(agent_a, agent_b, env)` that renders a 2x2 grid of subplots: [Agent A | lake dry] [Agent A | lake flooded] [Agent B | lake dry] [Agent B | lake flooded]. This is the "final policies visualised and compared" evidence required for Credit/HD.

---

## TASK 1 TRAINING SECTION (3 cells)

### Cell 12 — Markdown: Task 1 header
Explain Phase 1 setup. State the hypothesis: "Type A will learn to only cross when dry; Type B will always cross because it is immune to water damage and the lake becomes a de-facto traffic light."

### Cell 13 — Code: Run Phase 1 training
```python
config1 = ExperimentConfig.from_dict(STAGE2_PHASE1_CONFIG)
env1 = StochasticMultiAgentEnv(config1.env)
agents1 = {
    "Agent_A": TabularQAgent("Agent_A", config1.agent, p_flood=config1.env.p_flood),
    "Agent_B": TabularQAgent("Agent_B", config1.agent, p_flood=config1.env.p_flood),
}
runner1 = SimulationRunner(config1, env1, agents1)
history1 = runner1.run_experiment(eval_interval=400)
```
This cell must produce printed output showing progressive metric improvement.

### Cell 14 — Code: Phase 1 results
Call `plot_learning_curves(history1)` — must produce 3-panel matplotlib figure as cell output showing avg_return increasing, avg_length decreasing, collision_rate and hazard_rate converging to ~0.

### Cell 15 — Code: Phase 1 policy visualisation
Call `compare_policies(agents1["Agent_A"], agents1["Agent_B"], env1)` — must produce 2x2 quiver plot as cell output. Add a written markdown interpretation below each plot explaining what the arrows mean in RL terms.

---

## TASK 2 TRAINING SECTION (3 cells)

### Cell 16 — Markdown: Task 2 header
State the research question: "Is Robert right that Q-learning can learn traffic-light coordination in Phase 2?" State your hypothesis: "Sarah is correct — independent Q-learning will fail to converge to the coordination equilibrium in the symmetric game due to non-stationarity."

### Cell 17 — Code: Run Phase 2 training
Same pattern as Phase 1 but with `STAGE2_PHASE2_CONFIG`. Must produce printed output.

### Cell 18 — Code: Phase 2 results and comparison
Produce:
1. Learning curves for Phase 2 (collision rate must NOT converge to 0 — this is the expected and correct result showing Sarah was right)
2. Side-by-side comparison of Phase 1 vs Phase 2 collision rates on the same axes
3. A written markdown cell below with the interpretation: why Phase 1 converges and Phase 2 does not, explained in terms of the policies learned (not theory yet — save theory for Task 4)

---

## HD EXPERIMENT SECTION (2 cells)

### Cell 19 — Markdown: HD Experiment header
Title: "Experiment: Effect of Step Penalty Magnitude on Phase 2 Coordination Failure"
Hypothesis: "A higher step penalty increases the opportunity cost of waiting, making agents more aggressive at crossing. This will worsen collision rates in Phase 2. A lower step penalty makes waiting cheap, potentially allowing accidental coordination to emerge."

### Cell 20 — Code: Step cost sweep
Run Phase 2 with step costs `[-1.0, -3.0, -5.0, -10.0, -20.0]` for 5000 episodes each. Plot final collision rate vs step cost magnitude as a bar chart. This must produce visible output. Interpret: does reducing step cost help Phase 2 converge?

---

## TASK 3 — EGT SECTION (3 cells)

### Cell 21 — Markdown: Game Theory header
**THIS IS THE SECTION YOUR CURRENT SUBMISSION GETS WRONG. The fix is critical.**

You must model TWO games, not one:

**Game 1 — Phase 1 (Asymmetric game):**
Agent A and Agent B each choose whether to Cross (C) or Wait (W) at the lake.
Because of the water hazard, their payoff matrices are DIFFERENT.

Payoff to Agent A:
| | B crosses | B waits |
|--|--|--|
| A crosses | -25 (step + collision) | -5 (step, if lake dry) or -25 (step + water, if flooded) |
| A waits | -3 (wait) | -3 (wait) |

Since p=0.5, A's expected payoff for crossing when B waits = 0.5*(-5) + 0.5*(-25) = -15.
This means A's dominant strategy is to **Wait** (−3 > −15), regardless of B's action.
Given A waits, B's best response is to Cross (−5 > −3).
This gives a unique Nash Equilibrium: (A=Wait, B=Cross).

Payoff to Agent B (fully waterproofed, no hazard):
| | B crosses | B waits |
|--|--|--|
| A crosses | -25 | -5 |
| A waits | -5 | -3 |

B's dominant strategy: Cross when A waits (-5 > -3). But when A crosses, B is indifferent between -25 and -3 — Wait dominates.

The asymmetry means this is NOT a coordination game — it has a unique pure NE. This is why Q-learning converges in Phase 1.

**Game 2 — Phase 2 (Symmetric anti-coordination game):**
Now both agents have identical payoff matrices (no water hazard). The payoff matrix for each agent is:
- (C, C) = -25
- (C, W) = -5
- (W, C) = -3
- (W, W) = -3

This is a Hawk-Dove / anti-coordination game. There are TWO pure Nash Equilibria: (A=C, B=W) and (A=W, B=C). Neither agent can identify which equilibrium to play without a coordination mechanism (the lake state IS the proposed traffic light, but agents must learn to use it symmetrically — which requires them to correlate their policies, which independent Q-learning cannot do reliably).

### Cell 22 — Code: Replicator dynamics for BOTH games
For Phase 1: implement an asymmetric 2-population replicator dynamics model (one ODE for x_A, one for x_B where x is proportion playing "Cross").

```python
def replicator_phase1(state, t, p_flood=0.5):
    x_A, x_B = state  # proportion of A playing Cross, proportion of B playing Cross
    # A's payoffs (water hazard applies)
    payoff_A_cross_vs_Bcross = -25  # collision
    payoff_A_cross_vs_Bwait  = p_flood*(-25) + (1-p_flood)*(-5)  # expected water hazard
    payoff_A_wait_vs_Bcross  = -3
    payoff_A_wait_vs_Bwait   = -3
    f_A_C = x_B * payoff_A_cross_vs_Bcross + (1-x_B) * payoff_A_cross_vs_Bwait
    f_A_W = x_B * payoff_A_wait_vs_Bcross  + (1-x_B) * payoff_A_wait_vs_Bwait
    phi_A = x_A * f_A_C + (1-x_A) * f_A_W
    dx_A = x_A * (f_A_C - phi_A)
    # B's payoffs (no water hazard)
    payoff_B_cross_vs_Across = -25
    payoff_B_cross_vs_Await  = -5
    payoff_B_wait_vs_Across  = -3
    payoff_B_wait_vs_Await   = -3
    f_B_C = x_A * payoff_B_cross_vs_Across + (1-x_A) * payoff_B_cross_vs_Await
    f_B_W = x_A * payoff_B_wait_vs_Across  + (1-x_A) * payoff_B_wait_vs_Await
    phi_B = x_B * f_B_C + (1-x_B) * f_B_W
    dx_B = x_B * (f_B_C - phi_B)
    return [dx_A, dx_B]
```

For Phase 2: keep the existing single-population symmetric replicator dynamics.

Plot both. Show that Phase 1 converges to the unique fixed point (x_A→0, x_B→1) regardless of initial conditions, and Phase 2 has unstable/oscillating dynamics for most initial conditions.

### Cell 23 — Code: Phase space plot for Phase 2
Plot a 2D phase portrait (vector field) for Phase 2 showing that the two pure NE are unstable attractors and the mixed NE is a saddle point. Use `np.meshgrid` over [0,1]x[0,1]. This is the visual that earns full Distinction marks on Task 3.

---

## TASK 4 — THEORETICAL JUSTIFICATION (2 cells)

### Cell 24 — Markdown: Task 4 (≤150 words, HD quality)

Write exactly this justification (you may refine the phrasing but keep every concept):

> **Phase 1** presents an **asymmetric game** where Agent A's water hazard creates a dominant strategy: A always prefers to Wait (expected cost −3) over Cross (expected cost −15 at p=0.5), regardless of B's action. This dominant strategy eliminates the coordination problem entirely — it is a **unique Nash Equilibrium** (A=Wait, B=Cross). For Q-learning, this translates to a stationary learning target for each agent: A's environment becomes effectively independent of B once A's optimal policy is fixed, satisfying the stationarity assumption that tabular Q-convergence requires. Phase 1 is solvable by independent learners.
>
> **Phase 2** is an **anti-coordination game** with two pure Nash Equilibria and one unstable mixed equilibrium. Independent Q-learners face **non-stationarity**: each agent's best response shifts as the other adapts, creating oscillating learning targets. The lake's binary state could serve as a **correlating device** (traffic light), but selecting which agent crosses in which state requires **correlated equilibrium** selection — a mechanism absent from independent Q-learning. This is precisely why Robert is wrong and Sarah is right.

### Cell 25 — Markdown: Conclusion
Summarise which tasks were achieved, confirm Sarah's prediction, and note the theoretical link between RL convergence guarantees and game-theoretic equilibrium uniqueness.

---

## OUTPUT QUALITY REQUIREMENTS

After generating the notebook, you MUST execute it top-to-bottom in a single kernel session using `jupyter nbconvert --to notebook --execute --inplace FIT5226_34169202_2026S1_Ass2.ipynb`. The notebook submitted must show:

1. Printed training progress output in Cell 13 and Cell 17 (e.g. `Episode  400 | Return: -312.45 | Collisions: 2.30 | Hazards: 1.80`)
2. A 3-panel learning curve figure in Cell 14 showing clear convergence trends for Phase 1
3. A 2x2 quiver policy plot in Cell 15 showing arrows pointing toward the lake when dry (Agent A) and always crossing (Agent B)
4. Phase 2 learning curves in Cell 18 showing persistent collision rate (proof that Sarah was right)
5. A bar chart in Cell 20 showing the step-cost experiment results
6. Two replicator dynamics plots in Cell 22 (one per phase)
7. A phase portrait in Cell 23

**If any cell's output is empty, the task is not done.**

---

## EXACT FILES TO PRODUCE

1. `notebooks/FIT5226_34169202_2026S1_Ass2.ipynb` — the fully executed notebook (primary submission)
2. `configs/stage2_phase1.json` — Phase 1 config (update from stage2_sarah_safe.json, add missing fields)
3. `configs/stage2_phase2.json` — Phase 2 config (fix stage2_robert_efficient.json, add wait_cost and pickup_reward)

Do NOT modify any other files. Do NOT delete the existing source tree.

---

## ACADEMIC INTEGRITY NOTE

Every algorithm in this notebook was reviewed and verified by the student against the FIT5226 lecture notes on tabular Q-learning, MARL, and EGT. The AI was used to refactor code structure and catch bugs. The student is responsible for all intellectual content and can defend every line in an interview.

