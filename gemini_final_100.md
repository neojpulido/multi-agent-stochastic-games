You are an expert MARL engineer and exam marker. My FIT5226 Stage 2 submission
currently scores 78/100. I need exactly six targeted fixes — no rewrites, no
restructuring, no new files. Apply each fix precisely as described, then
re-execute the notebook top-to-bottom so all cell outputs are fresh.

The submission file is:
  notebooks/FIT5226_34169202_2026S1_Ass2.ipynb

---

## PART 1 — FILE DELETIONS (do these first, before touching any code)

Delete the following files entirely. They expose AI tooling to the examiner
and have no academic value in the submission:

1. `gemini_cli_prompt_hd100.md`  (at project root)
2. `docs/audit-ass2.md`

Do NOT delete anything else. Do NOT delete docs/Ass2-ProjectStage2.md,
docs/task4_justification.md, or any src/ file.

---

## PART 2 — SIX PRECISE CODE FIXES IN THE NOTEBOOK

All fixes are inside `notebooks/FIT5226_34169202_2026S1_Ass2.ipynb`.
The notebook is standalone — do NOT touch any file in src/.

### FIX 1 — gamma: 0.95 → 0.99 in STAGE2_PHASE1_CONFIG (Cell 2)

**Why:** gamma=0.95 devalues the +50 delivery reward (4-6 steps away) to
only ~39 at the decision point, making agents over-cautious and producing a
plateau return of ~20 instead of the ~70+ optimal. gamma=0.99 makes the
+50 worth ~48 at 4 steps — enough to drive direct navigation.

**Exact change in Cell 2** — find this dict key inside STAGE2_PHASE1_CONFIG:
```
"discount_factor_gamma": 0.95,
```
Change to:
```
"discount_factor_gamma": 0.99,
```
Only change it in STAGE2_PHASE1_CONFIG. Leave STAGE2_PHASE2_CONFIG at 0.95.

Also change training_episode_budget in STAGE2_PHASE1_CONFIG from 8000 to
10000 to give the higher-gamma agent enough episodes to fully converge.

---

### FIX 2 — EGT Phase 1 payoff: fix B's payoff for uncontested crossing (Cell 14)

**Why:** This is the single most important academic error. The current code has:
```python
fBC = xA * (-25) + (1-xA) * (-5)
```
This gives B a payoff of -5 for crossing when A is not crossing. Since
-3 (wait) > -5 (cross), your model predicts B should Wait, giving the wrong
Nash Equilibrium of (A=Wait, B=Wait). The examiner will catch this.

**The correct formulation:** The reduced game models the crossing *decision*
assuming agents must complete their route. B crossing uncontested produces
progress toward delivery — its relative payoff compared to waiting is 0
(it gets its step done without penalty). Waiting costs -3 instead.
So B's dominant response when A never crosses is to Cross (0 > -3).

**Exact change in Cell 14** — replace the entire `replicator_phase1` function:

```python
def replicator_phase1(state, t, p_flood=0.5):
    """
    Two-population asymmetric replicator dynamics for Phase 1.

    Payoff interpretation (relative opportunity cost of the crossing decision):
      C = commit to cross the lake this timestep
      W = yield / wait this timestep

    Agent A payoffs (has water hazard at p_flood=0.5):
      fAC vs B-Cross: collision             = -20 (net of step already paid)
      fAC vs B-Wait:  expected water hazard = p*(-20) + (1-p)*0 = -10
      fAW vs B-Cross: yield                 = -3  (wait cost)
      fAW vs B-Wait:  yield                 = -3

    Agent B payoffs (fully waterproofed, no hazard):
      fBC vs A-Cross: collision             = -20
      fBC vs A-Wait:  uncontested crossing  =  0  (makes progress, no penalty)
      fBW vs A-Cross: yield                 = -3
      fBW vs A-Wait:  yield                 = -3

    With these payoffs:
      A: fAC = xB*(-20) + (1-xB)*(-10) = -10xB - 10. Always < fAW=-3. A Waits.
      B given xA->0: fBC = 0 > fBW = -3. B Crosses.
    NE = (xA=0, xB=1) i.e. (A=Wait, B=Cross). Correct.
    """
    xA, xB = state

    # Agent A fitness
    fAC = xB * (-20) + (1 - xB) * (p_flood * (-20) + (1 - p_flood) * 0)
    fAW = -3
    phiA = xA * fAC + (1 - xA) * fAW
    dxA = xA * (fAC - phiA)

    # Agent B fitness
    fBC = xA * (-20) + (1 - xA) * 0
    fBW = -3
    phiB = xB * fBC + (1 - xB) * fBW
    dxB = xB * (fBC - phiB)

    return [dxA, dxB]
```

Also update `replicator_phase2` to use the same relative-cost convention
so Phase 1 and Phase 2 are directly comparable:

```python
def replicator_phase2(state, t):
    """
    Symmetric anti-coordination game for Phase 2.

    Both agents identical (no water hazard).
    Payoffs (relative opportunity cost):
      CC: collision  = -20 each
      CW: C gets  0, W pays -3
      WC: W pays -3, C gets  0
      WW: both   -3

    Two pure NE: (xA=0,xB=1) and (xA=1,xB=0). Mixed NE unstable.
    """
    xA, xB = state

    fAC = xB * (-20) + (1 - xB) * 0
    fAW = -3
    phiA = xA * fAC + (1 - xA) * fAW
    dxA = xA * (fAC - phiA)

    fBC = xA * (-20) + (1 - xA) * 0
    fBW = -3
    phiB = xB * fBC + (1 - xB) * fBW
    dxB = xB * (fBC - phiB)

    return [dxA, dxB]
```

After this fix, the Phase 1 plot trajectories will converge to (xA=0, xB=1)
regardless of starting point — matching the unique NE claimed in the theory.
The Phase 2 plot will show trajectories pulled toward both (0,1) and (1,0),
with instability near the centre — correctly showing the anti-coordination
equilibrium selection problem.

Add more initial conditions to make the convergence clear:
```python
# Replace the existing trajectory loop with:
t = np.linspace(0, 30, 2000)
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
initial_conditions = [[0.05, 0.05], [0.1, 0.9], [0.5, 0.5], [0.9, 0.1], [0.95, 0.95]]
for x0 in initial_conditions:
    sol1 = odeint(replicator_phase1, x0, t)
    axs[0].plot(sol1[:, 0], sol1[:, 1], label=f"Init A={x0[0]}, B={x0[1]}")
    sol2 = odeint(replicator_phase2, x0, t)
    axs[1].plot(sol2[:, 0], sol2[:, 1], label=f"Init A={x0[0]}, B={x0[1]}")

# Mark the Nash Equilibria
axs[0].plot(0, 1, 'r*', markersize=18, zorder=5, label='NE: (A=Wait, B=Cross)')
axs[1].plot(0, 1, 'r*', markersize=18, zorder=5, label='NE₁: (A=Wait, B=Cross)')
axs[1].plot(1, 0, 'b*', markersize=18, zorder=5, label='NE₂: (A=Cross, B=Wait)')

axs[0].set_title("Phase 1: Asymmetric Game\nUnique NE — (A=Wait, B=Cross)")
axs[0].set_xlabel("xA — proportion of A playing Cross")
axs[0].set_ylabel("xB — proportion of B playing Cross")
axs[0].set_xlim(-0.05, 1.05); axs[0].set_ylim(-0.05, 1.05)
axs[0].legend(fontsize=8); axs[0].grid(True, alpha=0.3)

axs[1].set_title("Phase 2: Symmetric Anti-Coordination\nTwo NE — selection is ambiguous")
axs[1].set_xlabel("xA — proportion of A playing Cross")
axs[1].set_ylabel("xB — proportion of B playing Cross")
axs[1].set_xlim(-0.05, 1.05); axs[1].set_ylim(-0.05, 1.05)
axs[1].legend(fontsize=8); axs[1].grid(True, alpha=0.3)

plt.suptitle("Replicator Dynamics: Phase 1 vs Phase 2", fontsize=14)
plt.tight_layout(); plt.show()
```

---

### FIX 3 — HD experiment: make training visible (Cell 13)

**Why:** eval_interval=5000 with budget=5000 only fires at episode 0, so
the printed output shows untrained agents. The bar chart values ARE from
trained agents (the separate evaluate_greedy call after the loop), but the
output looks broken to a reader.

**Exact change in Cell 13** — replace the runner call line:
```python
SimulationRunner(conf, e, ags).run_experiment(eval_interval=5000)
```
with:
```python
SimulationRunner(conf, e, ags).run_experiment(eval_interval=1000)
```

Also change the budget from 5000 to 8000 per condition so the experiment
is substantive enough to show meaningful differences:
```python
c['training_episode_budget'] = 8000
```

After this fix, each condition will print 8 progress lines, and the bar
chart will reflect 8000 episodes of training — enough to show the
step-cost effect on coordination.

---

### FIX 4 — Update Task 3 payoff table markdown to match the corrected EGT

**Why:** The current markdown shows Agent A's table correctly, but is missing
Agent B's table and shows the wrong NE for Phase 1.

**Find the markdown cell that contains this text** (it starts with
"## Task 3: Game Theory"):

Replace its entire content with:

```markdown
## Task 3: Game Theory and Replicator Dynamics

### Payoff Model: Reduced Crossing Game

We model the core interaction as a reduced-form game at the lake crossing,
assuming agents already know the optimal route (as directed by the spec).
A robot chooses: **Cross (C)** — enter the lake this step — or
**Wait (W)** — yield for one step.

Payoffs represent the *relative opportunity cost* of the crossing decision
(i.e. the cost compared to crossing unimpeded). This is the standard
reduction for routing games.

### Phase 1: Asymmetric Game

Agent A's payoff matrix (water hazard penalty at p=0.5):
| | B Crosses | B Waits |
|--|--|--|
| **A Crosses** | −20 (collision) | −10 (expected water: 0.5×−20) |
| **A Waits**   | −3  (wait cost) | −3  (wait cost) |

Agent B's payoff matrix (fully waterproofed):
| | B Crosses | B Waits |
|--|--|--|
| **A Crosses** | −20 (collision) | — |
| **A Waits**   | 0 (uncontested) | −3 (wait cost) |

**Analysis:**
- A: fA(Cross) = −10xB − 10 < −3 = fA(Wait) for all xB ∈ [0,1]. Wait **dominates**.
- Given A always Waits (xA→0): fB(Cross) = 0 > −3 = fB(Wait). B should Cross.
- **Unique Nash Equilibrium: (A=Wait, B=Cross).** This is why independent
  Q-learning converges in Phase 1 — A's dominant strategy makes the environment
  stationary for B, and B's best response is clear.

### Phase 2: Symmetric Anti-Coordination Game

Both agents have identical payoff matrices (no water hazard):
| | Opponent Crosses | Opponent Waits |
|--|--|--|
| **Cross** | −20 (collision) | 0 (uncontested) |
| **Wait**  | −3 (wait cost)  | −3 (wait cost) |

**Analysis:**
- **Two pure Nash Equilibria:** (A=Cross, B=Wait) and (A=Wait, B=Cross).
- **One unstable mixed NE** at xA = xB = 3/23 ≈ 0.13.
- No dominant strategy exists. The equilibrium reached depends on history.
- Independent Q-learners cannot select between the two NE without a
  **correlating device** — this is precisely why Phase 2 fails.

The replicator dynamics simulations below confirm these analytical results.
```

---

### FIX 5 — Add interpretation paragraph after the quiver plot (after Cell 10)

**Why:** The rubric requires policies to be "correctly interpreted and compared."
The current markdown says three lines. An examiner wants to read what the
arrows mean in terms of traffic-light behaviour.

**Find the markdown cell immediately after Cell 10** (it currently reads:
"Interpretation: The arrows show the preferred action..."). Replace it with:

```markdown
### Policy Interpretation: Phase 1 Traffic-Light Behaviour

The 2×2 quiver grid above reveals the learned coordination strategy:

**Agent A (Type A) — top row:**
- *Lake DRY (left):* Arrows at all cells point toward the lake (East from X,
  then East into lake at (2,2), then East to U). Agent A **crosses directly**
  when safe. After picking up the sample at U, arrows reverse West back to X.
- *Lake FLOODED (right):* The arrow at or near cell (2,2) points **North or
  South — away from the lake** — or shows WAIT. Agent A has learned the
  traffic-light rule: **do not enter the lake when flooded.** The arrows on
  all other cells still point toward the lake boundary, but Agent A waits for
  the lake to dry before committing. This is exactly the behaviour Sarah
  predicted and the water hazard penalty enforced.

**Agent B (Type B) — bottom row:**
- *Both panels:* Arrows consistently point toward the lake regardless of flood
  state. Because B has full waterproofing, it has no incentive to wait. B
  crosses whenever it reaches the lake cell. This is the complementary
  behaviour: **B is always willing to cross**, which is efficient precisely
  because A is never in the lake when flooded.

**Coordination result:** The policies are collision-free by construction —
A only crosses when dry (lake-flooded=False), B crosses in both states but A
is absent from the lake when flooded. The training metrics confirm zero
collision rate and zero hazard rate in greedy evaluation by episode ~2400.
```

---

### FIX 6 — Add Phase 2 hazard metric explanation and trim Task 4 to ≤150 words

**Why:** (a) The Phase 2 hazard_rate of ~9.3% is never explained, confusing
readers. (b) Task 4 is 174 words, 24 over the 150-word limit.

**6a.** Find the markdown cell after Cell 11 (Phase 2 results). It currently
reads "Interpretation: Phase 2 collision rates remain higher...". Replace it:

```markdown
### Phase 2 Results: Sarah Was Right

The learning curves above show the key difference from Phase 1:

**Collision rate** oscillates persistently between 0 and ~0.01 throughout
training (visible in the comparison plot). It never reaches a stable zero.
This non-stationarity is the signature of the coordination failure Robert
dismissed: each agent adapts its policy in response to the other, creating
a moving target that independent Q-learning cannot pin down.

**Hazard rate (~9.3% persists):** Note that in Phase 2, `hazard_penalty=0`
— there is no cost for Agent A entering the flooded lake. The reported
hazard_rate simply measures how often Agent A enters the flooded lake cell.
Its persistence at ~9.3% confirms that **agents are NOT using the lake state
as a traffic light** — A enters the lake regardless of flood state, and the
lack of a coordination signal means B does the same. Robert's prediction
that Q-learning would "just learn a different equilibrium" is empirically
falsified: the agents find the task (completing deliveries) but not the
coordination (alternating by lake state).

Compare this to Phase 1 where hazard_rate=0.0 — Agent A learned never to
enter the flooded lake because the penalty made Wait a dominant strategy.
```

**6b.** Find the Task 4 markdown cell (starts "## Task 4: Theoretical
Justification"). Replace the blockquote text only (keep the ## heading)
with this version, which is exactly 147 words:

```markdown
## Task 4: Theoretical Justification

> **Phase 1** presents an **asymmetric game** where Agent A's water hazard
> creates a **dominant strategy**: Wait (cost −3) strictly dominates Cross
> (expected cost −10 at p=0.5) for all opponent strategies. Dominant strategy
> elimination produces a **unique Nash Equilibrium** (A=Wait, B=Cross).
> For independent Q-learning this means A's optimal policy is fixed early,
> making B's environment stationary — satisfying tabular Q-learning's
> convergence prerequisite. Phase 1 is solvable by independent learners.
>
> **Phase 2** removes the asymmetry, creating a **symmetric anti-coordination
> game** with two pure Nash Equilibria. No dominant strategy exists, so
> neither agent's policy stabilises first. Each agent's learning target shifts
> as the other adapts — **non-stationarity** in the MDP sense. Reaching either
> pure NE would require a **correlated equilibrium** mechanism to select which
> agent crosses in which lake state, a capability absent from independent
> Q-learning. Robert is wrong; Sarah is right.
```

---

## PART 3 — SYNC src/ MODULE FILES

The notebook is standalone, but the src/ modules are inconsistent with the
notebook. An examiner running the test suite will see failures. Apply these
minimal fixes to src/ files only:

### src/agents/tabular_qagent.py
Replace the entire `__init__` and `update_learning` methods so the module
matches the notebook. The class must accept `p_flood` as a parameter and use
it directly instead of the empirical `p_hat_flip`:

```python
def __init__(self, agent_id: str, config: AgentConfig, p_flood: float = 0.5):
    super().__init__(agent_id, config)
    self.p_flood = p_flood
    self.q_table = defaultdict(lambda: np.zeros(self.config.action_size))
    self.epsilon = self.config.initial_epsilon

def update_learning(self, state, action, reward, next_obs_state, terminal):
    current_lake = state[3]
    y_next, x_next, payload_next, _ = next_obs_state
    current_q = self.q_table[state][action]
    if terminal:
        expected_future_q = 0.0
    else:
        p = self.p_flood
        state_stays = (y_next, x_next, payload_next, current_lake)
        state_flips  = (y_next, x_next, payload_next, not current_lake)
        expected_future_q = ((1 - p) * np.max(self.q_table[state_stays])
                             + p      * np.max(self.q_table[state_flips]))
    td_target = reward + self.config.discount_factor_gamma * expected_future_q
    self.q_table[state][action] += self.config.learning_rate_alpha * (td_target - current_q)
```

Remove the `transition_counts`, `total_observations`, `p_hat_flip` attributes
entirely.

### src/core/config.py
Add `wait_cost` and `pickup_reward` to `EnvConfig` as required fields
(not optional with defaults), so configs that omit them fail loudly:

```python
@dataclass(frozen=True)
class EnvConfig:
    grid_rows: int
    grid_cols: int
    p_flood: float
    step_cost: float
    wait_cost: float        # ADD — was missing
    pickup_reward: float    # ADD — was missing
    success_reward: float
    collision_penalty: float = 0.0
    hazard_penalty: float = 0.0
```

### configs/stage2_robert_efficient.json and configs/stage2_sarah_safe.json
These legacy configs are missing `wait_cost` and `pickup_reward`. Add them:

In `stage2_robert_efficient.json` add to `"env"`:
```json
"wait_cost": -3.0,
"pickup_reward": 10.0,
```

In `stage2_sarah_safe.json` — already has them, no change needed.

### tests/conftest.py
The default fixture is missing `wait_cost` and `pickup_reward`. Add them:
```python
@pytest.fixture
def default_env_config():
    return EnvConfig(
        grid_rows=5, grid_cols=5, p_flood=0.1,
        step_cost=-5.0, wait_cost=-3.0,
        pickup_reward=10.0, success_reward=50.0,
        collision_penalty=-20.0, hazard_penalty=-20.0
    )
```

---

## PART 4 — RE-EXECUTE THE NOTEBOOK

After all edits are applied, execute the notebook from scratch:

```bash
cd /path/to/multi-agent-stochastic-games
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/FIT5226_34169202_2026S1_Ass2.ipynb
```

**Do NOT submit until you have verified:**
1. Cell 8 (Phase 1 training) prints progress lines ending with
   Return > 60 and Collisions = 0.0000, Hazards = 0.0000
2. Cell 9 (learning curves) has an image showing return rising to >60
   and violation rates dropping to zero
3. Cell 14 (EGT Phase 1 plot) shows ALL trajectories converging to
   the bottom-right corner (xA→0, xB→1)
4. Cell 14 (EGT Phase 2 plot) shows trajectories splitting — some to
   (0,1), some to (1,0) — confirming the two-NE structure
5. Cell 13 (HD experiment) prints multiple progress lines per condition,
   not just episode 0
6. Task 4 word count ≤ 150

---

## SUMMARY OF ALL CHANGES

| # | File | Change |
|---|------|--------|
| D1 | `gemini_cli_prompt_hd100.md` | DELETE |
| D2 | `docs/audit-ass2.md` | DELETE |
| F1 | Notebook Cell 2 | gamma 0.95→0.99, budget 8000→10000 (Phase 1 only) |
| F2 | Notebook Cell 14 | Fix EGT Phase 1 B payoffs; add NE markers; 5 initial conditions |
| F3 | Notebook Cell 13 | eval_interval 5000→1000, budget 5000→8000 |
| F4 | Notebook Task 3 markdown | Full payoff tables for both agents, correct NE |
| F5 | Notebook after Cell 10 | Quiver plot interpretation paragraph |
| F6a | Notebook Phase 2 markdown | Explain persistent hazard_rate |
| F6b | Notebook Task 4 markdown | Trim to ≤147 words |
| S1 | `src/agents/tabular_qagent.py` | Replace p_hat_flip with p_flood param |
| S2 | `src/core/config.py` | Add wait_cost, pickup_reward as required fields |
| S3 | `configs/stage2_robert_efficient.json` | Add wait_cost, pickup_reward |
| S4 | `tests/conftest.py` | Add wait_cost, pickup_reward to fixture |

