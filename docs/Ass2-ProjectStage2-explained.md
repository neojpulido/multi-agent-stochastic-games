# FIT5226 Project 
## Stage 2 - Multi-agent coordination
This document describes Stage 2 of the FIT 5226 project, the final stage. We build on the work done in Stage 1. If you didn’t manage to solve Stage 1, a Master solution will be made available on the Moodle site for you to use as a foundation.

| Assessment | Topic | Release | Due | Weight | Assessment mode |
| --- | --- | --- | --- | --- | --- |
| **in-semester** | Project Stage 1 (Table-based RL, single agent) | Wk3 | April 11 | 15% | Individual |
| **in-semester** | Project Stage 2 (MARL coordination task) | Wk 6 | May 15 | 35% | Individual |
| **exam** | All topics but no programming |  |  | 50% |  |


This project constitutes your entire in-semester assessment for the unit.

Stage 2 (the current Phase) concerns Multi-agent coordination. 

## Scenario for Stage 2
You will write code for multiple agents of two different types to perform q-learning for a coordination task in a small grid world. The emphasis in this assignment is not on the coding itself—this should be straightforward after Stage 1. The emphasis is not even on successfully learning all aspects of all tasks. Your agents may or may not be able to learn the more advanced aspects of the scenario but this is not what determines success in this assignment. ***The emphasis is on understanding how RL behaves in this scenario and what makes one variant of the problem harder than the other one. For a high level of achievement we expect you to systematically use structured experimentation to find the answer to this and/or to be able to explain this in a theoretically grounded framework***.

Here is the scenario. Read it carefully, the details are important!

The space agency has contracted your company to design autonomous robots for sample collection in rugged terrain on a remote planet. 

The mission plan designates four relevant locations: The landing sites of two spaceships (X in the west, Y in the north) and two target sites (U in the east, V in the south), where we hope to find evidence of life. The robots will be deployed from the spaceships at X and Y and their task is to travel to U and V, retrieve soil samples and come back to the same ship they were deployed from. Robots deployed at X have to sample at U and robots deployed at Y have to sample at V. The locations of X, Y, U and V are fixed as shown in the diagram.

**Implementation Mapping:**
```python
# src/domain/gridworld/stochastic.py

# 1. Fixed Coordinate Landmarks
self.locs = {
    "X": (2, 0), "Y": (0, 2), "U": (2, 4), "V": (4, 2), "Lake": (2, 2)
}

# 2. Heterogeneous Deployment (Initial State)
def reset(self):
    self.positions = {"Agent_A": self.locs["X"], "Agent_B": self.locs["Y"]}
    self.has_sample = {"Agent_A": False, "Agent_B": False}

# 3. Task Progress & Sparse Rewards (Search -> Sampling -> Return -> Delivery)
for agent_id in ["Agent_A", "Agent_B"]:
    if atype == "Type_A":
        if not self.has_sample[agent_id] and pos == self.locs["U"]:
            self.has_sample[agent_id] = True; rewards[agent_id] += 10.0 # Sampled
        elif self.has_sample[agent_id] and pos == self.locs["X"]:
            self.done[agent_id] = True; rewards[agent_id] += 50.0   # Delivered
    else: # Type_B
        if not self.has_sample[agent_id] and pos == self.locs["V"]:
            self.has_sample[agent_id] = True; rewards[agent_id] += 10.0 # Sampled
        elif self.has_sample[agent_id] and pos == self.locs["Y"]:
            self.done[agent_id] = True; rewards[agent_id] += 50.0   # Delivered
```

**Technical Explanation:**
> The implementation operationalizes the mission plan as a **Stochastic Transport Task** with the following engineering constraints:
> 1.  **Heterogeneous Topology:** The environment distinguishes between `Agent_A` and `Agent_B` objectives. By mapping spaceship $X$ to target $U$ and $Y$ to $V$, the simulation creates two orthogonal task axes that intersect at the lake $(2,2)$.
> 2.  **State-Driven Logistics:** Mission progress is tracked via a `has_sample` state variable. This enforces a strictly sequential task dependency: Search $\rightarrow$ Sampling $\rightarrow$ Return $\rightarrow$ Delivery.
> 3.  **Incentive Alignment:** Agents are not given explicit goal coordinates; instead, the environment provides sparse positive reward signals (10 for sampling, 50 for delivery). The Q-learning algorithm must discover the landmark locations by maximizing these signals.
> 4.  **Terminal Reward Coupling:** The 'delivery' reward is only accessible if the agent is in the 'return' state (`has_sample=True`), ensuring that agents learn to complete the entire transport cycle rather than just reaching the sampling site.

**Math Explanation:**
> The environment is a discrete grid $G = \{0, \dots, 4\} \times \{0, \dots, 4\}$. The shortest path distance between any two points $p_1, p_2 \in G$ is defined by the **Manhattan Distance**:
> $$L_1(p_1, p_2) = |y_1 - y_2| + |x_1 - x_2|$$
> The landmarks are strategically positioned such that their shortest paths intersect at a single point $P_{lake}$:
> - Path A ($X \rightarrow U$): $\{(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)\}$
> - Path B ($Y \rightarrow V$): $\{(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)\}$
> - Intersection: $Path_A \cap Path_B = \{(2, 2)\}$

We don’t know much about the planet except (1) that the terrain is dangerous and movement is extremely difficult, so that it is important that all routes have to be kept as short as possible; (2) there is a mysterious playa lake (an ephemeral lake) at the point where the shortest connections between X and U and Y and V, respectively, intersect. This lake floods and dries out at surprisingly high speed (but this may not be so strange given that this is a very strange planet anyway). 

**Implementation Mapping:**
```python
# src/domain/gridworld/stochastic.py

# 1. Incentivizing Shortest Paths (Dangerous Terrain)
if action == Directions.WAIT:
    rewards[agent_id] += self.config.step_cost * 0.6 # Wait: -3
else:
    rewards[agent_id] += self.config.step_cost       # Move: -5

# 2. Ephemeral Lake Stochasticity
if random.random() < self.config.p_flood:
    self.lake_flooded = not self.lake_flooded

# 3. Intersection Geometry
# Lake at (2,2) is the intersection of X->U (Row 2) and Y->V (Col 2)
self.locs["Lake"] = (2, 2)
```

**Technical Explanation:**
> The environment design reflects the planet's constraints through two primary mechanisms:
> 1.  **Cost-Based Path Optimization:** The 'dangerous terrain' is modeled via a continuous negative reward signal (`step_cost`). This ensures that the RL objective (maximizing cumulative reward) is mathematically equivalent to minimizing the Manhattan distance between mission landmarks.
> 2.  **Exogenous Stochasticity:** The lake's flooding behavior is implemented as a binary state machine that transitions independently of agent behavior. This introduces **State Non-Stationarity** from the agent's perspective, as the optimal action at the intersection $(2,2)$ depends on the stochastic `lake_flooded` signal.
> 3.  **Conflict Geometry:** The lake is strategically placed at the intersection of the agents' orthogonal shortest paths. This forces the agents to solve an **Anti-Coordination Game** to avoid collisions while pursuing their primary mission objectives.

**Math Explanation:**
> 1.  **Expected Step Reward:** The RL agent seeks to maximize $G_t = \sum \gamma^k r_{t+k+1}$. With $r_{step} = -5$, any deviation from the geodesic path incurs a penalty of at least $-5$ per extra step.
> 2.  **State Transition Kernel:** The lake state $s_{lake} \in \{0, 1\}$ evolves according to a Markov transition matrix $T$:
> $$ T = \begin{pmatrix} 1-p & p \\ p & 1-p \end{pmatrix} $$
> where $p$ is the transition probability. The global state transition is thus $P(s' | s, a) = P(s'_{agents} | s_{agents}, a) \cdot T(s'_{lake} | s_{lake})$.


As the head of the engineering department, you decide to build swarms of tiny robots, so that an almost continuous stream of robots can run between each ship and its target sampling location. 

Your project experiences two phases as another department meddles in it.

### Phase 1:

Robert, the head of the cost-cutting department wants your job, so he tries to sabotage your department by deciding that only half of the robots can be built with reliable waterproofing. ***The robots deployed at X (let’s call them Type A) only get a cheap and somewhat dodgy waterproofing so that there is a serious risk of damage if they attempt to cross the lake when it is flooded***. ***The ones to be deployed at Y (let’s call them Type B) get full waterproofing and they can cross the lake safely even if it is flooded***. At least Robert allocates some budget equipping each robot with a **water sensor** that indicates whether the lake is flooded. 

Importantly, water damage is not the only danger on the lake. Since it is also the crossing of the shortest paths for the two robot types and you have a more or less continuous stream of robots, ***there is a high danger of collision between robots of different types***, which would seriously damage them (robots of the same type are assumed not to collide; they are built to evade each other). 

Luckily, Sarah, your smartest engineer, saves the project by advising that she can equip the robots with reinforcement learning so that type A can learn to adjust its crossing behaviour according to the risk of water damage and both types can learn to avoid collisions.

**Implementation Mapping:**
```python
# src/domain/gridworld/stochastic.py

# 1. Asymmetric Hazard Penalty (Water Damage)
if self.lake_flooded and self.positions["Agent_A"] == self.locs["Lake"]:
    rewards["Agent_A"] += self.config.hazard_penalty # Only Type A at (2,2)

# 2. Universal Collision Penalty
if self.positions["Agent_A"] == self.positions["Agent_B"]:
    rewards["Agent_A"] += self.config.collision_penalty
    rewards["Agent_B"] += self.config.collision_penalty
```

**Technical Explanation:**
> Phase 1 is implemented as an **Asymmetric Game** where agents face different risk profiles:
> 1.  **Dodgy Waterproofing (Agent A):** The environment logic specifically targets `Agent_A` with a `hazard_penalty` when it occupies the lake cell $(2,2)$ while `lake_flooded` is True. This creates an immediate negative reward bias that discourages crossing during floods.
> 2.  **Full Waterproofing (Agent B):** `Agent_B` logic excludes the hazard check, allowing it to cross the flooded lake without environmental penalty.
> 3.  **Environmental Signal:** The water sensor (lake state) acts as a **Correlating Device**. Because Agent A is forced to wait when the lake is flooded to avoid damage, Agent B can learn that the flooded state is a "safe window" to cross without risk of collision, as Agent A is statistically unlikely to be there.
> 4.  **Collision Physics:** Regardless of waterproofing, both agents suffer a `collision_penalty` if they occupy the same coordinate. This reinforces the need for coordination even when environmental hazards are absent for one player.

**Math Explanation:**
> In Phase 1, the Reward Functions $R_A$ and $R_B$ are asymmetric at $s = (2,2)$:
> $$ R_A(s, flooded, a) = r_{step} + r_{hazard} \cdot \mathbb{1}(flooded) $$
> $$ R_B(s, flooded, a) = r_{step} $$
> This asymmetry creates a **Correlated Equilibrium** where the lake state $flooded$ serves as a public signal $\nu$. Since $E[R_A | flooded, cross] \ll E[R_A | flooded, wait]$, Agent A's optimal policy $\pi_A$ converges to $wait$. This allows Agent B to safely select $cross$, resolving the anti-coordination problem.


### Phase 2
 
Realising that he won’t succeed in sabotaging your project because of Sarah’s skills, Robert changes tack and advises that the space agency doesn’t fully trust her method. In an attempt to appear genuine, he now allows you to fully waterproof all robots of both types. Since there is no danger of water damage anymore, Robert decrees that the robots must no longer be penalised for entering the flooded lake so that they can figure out the optimal behaviour. However, the collision danger in the lake remains.

Robert declares that your reinforcement learning should still be able to avoid collisions by learning that one kind of robot should only cross when the lake is dry and the other kind should only cross when the lake is flooded. In some sense they should learn to use the lake as a traffic light. For reasons that remain unclear, you are not allowed to program their behaviour explicitly in this way but they have to learn this on the job. 

Robert cunningly argues that this should indeed be quite simple since, as he says, “using the lake like a traffic light is just a different equilibrium of the collective behaviour so they should learn this.” 

Sarah warns you that this will be extremely difficult to achieve using q-learning and that she has serious doubts the project will succeed under these conditions.



## Tasks

Note that you don’t have to address all tasks listed below for this assignment. For details which tasks have to be solved for which grade level, please refer to the indicative rubric at the end of this document. 

As the head of engineering, it is your responsibility to figure out how the project can be conducted successfully. Is Sarah right? Can you outsmart Robert?

You decide to build a small simulation to support your decision making. You set up a 5x5 grid-world as in the diagram and populate it with the two agent types, A and B. Both types have the same five actions that they can execute, namely to step to any neighboring field in the four cardinal directions (north, south, east, west) or to wait in the current position without moving. Type A starts at X and when it reaches location U it automatically picks up a sample. When it has returned to X it automatically discharges the sample. At this point, it has completed its task. Likewise, type B, deployed from Y, automatically picks up a sample when it reaches V and automatically discharges the sample when it returns to Y. No specific action is required from a robot for picking up or dropping an item, it only needs to step into the corresponding location. 

The task each robot has to learn is to find the shortest way to their designated sampling location and to avoid collisions and water damage. Each robot can observe its own location, whether it carries a sample, and the binary state of the lake (dry/flooded) but not the other agents’ locations. 

**Implementation Mapping:**
```python
# src/core/state.py
class StateHandler:
    @staticmethod
    def get_agent_state(agent_id, observation):
        pos = observation["positions"][agent_id]
        has_sample = observation["has_sample"][agent_id]
        lake_flooded = observation["lake_flooded"]
        return (pos[0], pos[1], has_sample, lake_flooded)
```

**Technical Explanation:**
> To comply with the *partial observability* requirement, the `StateHandler` filters the global environment observation. Each agent only receives its own (y, x) coordinates, its payload status, and the global lake state. The other agent's position is explicitly excluded to ensure decentralized learning. The resulting tuple is used as a hashable key for the tabular Q-table.

##3Task 1:

Sarah’s theory about Phase 1 is correct. The robots can learn that Type A should only cross when the lake is dry and Type B consequently should only enter the lake when it is flooded so that the crossing is guaranteed to be collision-free  (as there will be no Type A in the lake at that time). 

Build a simulation to demonstrate that the agents will learn this behaviour with q-learning. 

At each time all agents step simultaneously. This means that an agent sees the results of the actions by all other agents in the same time step. For example, if two agents stand next to the lake at time t and both decide to step into the lake, they will collide in time step t. Agents only see the final result of these steps, so if an A agent is in location X at time t and a B agent moves to X at t+1 while the a agent moves away from X at t+1 they do not collide. 

**Implementation Mapping:**
```python
# src/domain/gridworld/stochastic.py
def step(self, joint_action):
    # Simultaneous Movement Resolution
    prev_positions = self.positions.copy()
    new_positions = {}
    for agent_id, action in joint_action.items():
        potential_pos = Actions.apply_action(prev_positions[agent_id], action)
        new_positions[agent_id] = potential_pos if is_valid(potential_pos) else prev_positions[agent_id]
    self.positions = new_positions # Update all at once
```

**Technical Explanation:**
> The environment resolves movements in a *simultaneous lock-step* fashion. Instead of updating agents sequentially (which would give the second agent an unfair informational advantage), the `step` function captures the `prev_positions`, calculates all `new_positions` independently, and then commits them. Collisions are then checked based on these finalized resulting coordinates.

The state of the lake is updated in-between successive time steps according to a probability p that determines the probability of it changing its state from dry to flooded (or vice versa) at this moment.

**Implementation Mapping:**
```python
# src/domain/gridworld/stochastic.py
if random.random() < self.config.p_flood:
    self.lake_flooded = not self.lake_flooded
```

**Technical Explanation:**
> Stochastic transitions are handled before movement resolution in the `step` loop. This ensures that the rewards calculated (e.g., water hazard penalty) reflect the state of the environment *during* the action execution.

Your risk-assessment engineers advise that the following rewards adequately reflect the cost and likelihood of damage. You may modify these but they provide a good starting point: 

Step = -5, Wait = -3, Collision = -20, Type A entering water = -20

Of course, these scale with the other rewards and the above is valid for sample-location-reached = 10, sample-delivered = 50.

**Implementation Mapping:**
```json
// configs/stage2_sarah_safe.json
"env": {
    "step_cost": -5.0,
    "success_reward": 50.0,
    "collision_penalty": -20.0,
    "hazard_penalty": -20.0
}
```

**Technical Explanation:**
> Hyperparameters and reward values are externalized into JSON configuration files. This allows for *modular experimentation* (required for the HD grade) without modifying core algorithmic code. The `hazard_penalty` is set to `-20.0` for Phase 1 and `0.0` for Phase 2.

***Hints:***

1. We strongly advise using tabular Q-Learning and not Deep-Q. While Deep-Q will work, it is absolute overkill, can get quite messy to set up correctly, and it blows the learning time out of proportion.
2. As in the first assignment, no additional heuristics or trickery with the rewards (encoding heuristics in the reward schema) are permissible. 
3. Learning this task is quite sensitive to the setting of the discount factor ɣ. 
4. When your simulation at first does not succeed, Sarah advises that it is very important that you account for both possible states of the lake in the next state. It could dry out or flood at any point of time. Make sure to follow her advice. 

**Implementation Mapping:**
```python
# src/agents/tabular_qagent.py
def update_learning(self, state, action, reward, next_obs_state, terminal):
    # Sarah's Expected Value Bellman Update
    expected_future_q = ((1 - p) * max_q_unchanged) + (p * max_q_flipped)
    td_target = reward + (gamma * expected_future_q)
    self.q_table[state][action] += alpha * (td_target - current_q)
```

**Technical Explanation:**
> Sarah's update is a *Model-Aware Expected Value Bellman Update*. Instead of a standard TD-target using only the observed next state, we use the known transition probability $p$ to weight the maximum future Q-values of both possible next states (lake flooded vs dry). This stabilizes the Q-values in a highly stochastic environment.

**Math Explanation:**
> Sarah's algorithm is a variation of **Full-Backup Q-Learning**. While standard Q-learning samples a single next state $s' \sim P(s'|s,a)$, Sarah's update analytically computes the expectation over the known transition model of the lake:
> $$ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \sum_{s' \in S} P(s'|s,a) \max_{a'} Q(s', a') - Q(s, a) \right] $$
> In our specific case, with binary lake states $L \in \{dry, flooded\}$ and transition probability $p$:
> $$ \mathbb{E}[V(s')] = (1-p) \cdot \max_{a'} Q(s'_{dry}, a') + p \cdot \max_{a'} Q(s'_{flooded}, a') $$
> This reduction in **Target Variance** significantly accelerates convergence in stochastic environments by smoothing out the noise from exogenous state flips.

5. You are allowed to split the simulation into multiple learning phases instead of just a single integrated simulation in which both types learn in parallel. However, you may only do so if you can provide a sound argument why this is equivalent to letting them learn in an integrated way and on-the-job.
6. Please read the section about generative AI use at the end of this document very carefully. While you are allowed and even encouraged to use AI for coding and learning purposes, you must be able to explain the submitted product completely yourself. This applies to all tasks.


### Task 2:

Build a simulation to investigate whether Sarah or Robert is right regarding Phase 2. 
Find out what happens in the changed scenario and analyse the outcomes.

### Task 3:

Back up your insights and observations from Task 1 and Task 2 from a game-theoretic perspective. Reduce the whole scenario to the core crossing decision (i.e. assume that the robots already know how to navigate the shortest path). Devise a game that reflects this situation meaningfully and use replicator dynamics to find out what the behaviour converges on. You are free to simulate replicator dynamics numerically or to solve it analytically as you see fit.

**Implementation Mapping:**
```python
# src/math/egt_model.py
def replicator_dynamics(x, t):
    # dx/dt = x * (f_C - phi)
    f_C = x * (-25) + (1 - x) * (-5)
    f_W = x * (-3) + (1 - x) * (-3)
    phi = x * f_C + (1 - x) * f_W
    return x * (f_C - phi)
```

**Technical Explanation:**
> The intersection conflict is abstracted into a *Symmetric Normal-Form Game*. We define a payoff matrix based on the environment's rewards (Collision vs Delay). Using `scipy.integrate.odeint`, we numerically solve the Replicator Dynamics ODE to demonstrate that "Cross" is not an Evolutionarily Stable Strategy (ESS) in the symmetric Phase 2.

**Math Explanation:**
> We model the intersection as a symmetric 2x2 game with strategies $C$ (Cross) and $W$ (Wait). The **Replicator Equation** governs the evolution of the population proportion $x$ playing strategy $C$:
> $$ \dot{x} = x(f_C(x) - \bar{f}(x)) $$
> where $f_C$ is the fitness of crossing and $\bar{f}$ is the mean population fitness.
> **Payoff Matrix $A$ (Phase 2):**
> $$ A = \begin{pmatrix} -25 & -5 \\ -3 & -3 \end{pmatrix} $$
> - $U(C, C) = -25$ (Collision + Step)
> - $U(C, W) = -5$ (Step only)
> - $U(W, C) = -3$ (Wait only)
> - $U(W, W) = -3$ (Mutual Wait)
> Since $U(W, C) > U(C, C)$, the 'Cross' strategy is unstable against 'Wait' mutants when the population is mostly 'Crossers'. The simulation tracks the convergence toward the stable Nash Equilibrium.

### Task 4 (max 150 words)

Explain your findings from the previous tasks from a theoretical perspective. In other words, give a conceptual argument that is firmly grounded in the theory of reinforcement learning and game theory that explains why the outcomes observed in Task 1-3 make sense.

**Implementation Mapping:**
```markdown
# docs/task4_justification.md
"In Phase 1, the water hazard naturally imposes an asymmetric game... 
facilitating clear equilibrium selection... However, in Phase 2, the removal 
of the water penalty creates a perfectly symmetric game."
```

**Technical Explanation:**
This section provides the *theoretical synthesis*. It explains that the water hazard in Phase 1 acts as an *environmental correlating device*, breaking the symmetry between agents and facilitating coordination. In Phase 2, the lack of such a device leads to non-stationarity and coordination failure.

 

## Submission Instructions

Submission will be via the Moodle platform. Detailed submission instructions will be published on Moodle in the Assignment section.

## Use of Generative AI

You are allowed and, in fact, encouraged to use Generative AI to solve your assignment. If you decide to do so, you must treat the AI like another external author (as a non-authoritative author whom you mistrust, given how much content is made up by Chat GPT and similar AIs). It is entirely your own responsibility that the content of your submission is correct and you can only use generated content to the extent that you could use materials provided by an external author. The AI is not part of your project team. If you use code or insights that an AI provided you must be able to fully explain every detail of it.

If you submit code that you are unable to explain in an interview, this will attract zero marks. 

You must give a declaration that fully explains how and for which components you used generative AI.

Any use of generative AI must be appropriately acknowledged (see Learn HQ).


## Indicative Rubric

Your solution will be marked on 
* Correctness but not elegance of the implementation 
* Design of the evaluation metrics
* Correct and insightful interpretation of the outcomes
* Clarity of the discussion and explanations
* Well-designed experimentation where used and/or required
* Level of linking observations meaningfully to theory

### Pass Level (Task 1)

You have implemented q-learning correctly for at least the first phase (Task 1). Your implementation clearly learns (but maybe not to optimality). You have defined meaningful metrics and demonstrated the learning using these metrics. Please familiarise yourself with the requirement to be able to fully explain all codes and outputs as outlined under “Use of Generative AI” above.

### Credit Level (Task 1 & 2)

You have implemented both phases correctly (minor errors may be present) and your implementation learns the first case to optimality (Task 1 and Task 2). Convincing evidence for full learning is given in the first case using the metrics defined. The final policies learned for both cases have been visualised, correctly interpreted and compared. The relationship between the solution qualities in both phases has been explained in terms of the policies learned. 

### Distinction Level (Task 1-3)

Requirements for credit plus either of the following

* 1) A solid attempt has been made at using GT/EGT to back up the reasoning presented (Task 3). This attempt shows a good understanding of GT/EGT and its relation to RL, relates meaningfully to the problem and is mostly complete but it may contain minor mistakes and/or omissions. 

Or

* 2) At least one meaningful experiment has been designed, conducted and discussed to analyse the harder nature of Phase 2. The discussion shows a good understanding of the harder nature of the second problem. We normally expect such experiments to proceed by modifying a particular structural aspect of the problem to reveal how this aspect impacts the learning. The experiment must be designed and outcomes of the experiment are meaningfully explained and analysed. 
  
  One interesting variation to investigate is : How do different levels of step penalties influence the outcome in Phase 2? 
  
  You are allowed to use this suggestion but you are free to design other variations if they reveal important aspects of the behaviour. 

### High Distinction Level (Task 1-4)

Both phases are implemented correctly and compared as simulations. At least the simulation of the first phase learns optimal behaviour. The simulation outcomes are observed correctly and documented using the metrics defined. 

The difference between the hardness of the problems for RL is explained with arguments that are solidly grounded in theoretical concepts of Reinforcement Learning and Game Theory (Task 4). They show a comprehensive understanding of the different characteristics of the problems for RL and their relation to game theory.

Theoretical explanations must be correct but only need to be given conceptually in broad sketch terms, i.e. they may but do not have to contain detailed mathematical working out. 

Don’t let Robert win! Ideally, you have found a way to outsmart Robert, maybe with Sarah’s help, but this is not a part of the requirements for a HD.




