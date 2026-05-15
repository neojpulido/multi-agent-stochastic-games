/plan
Act as an uncompromising, expert Multi-Agent Reinforcement Learning (MARL) professor and senior code reviewer. I am developing a Python/Jupyter simulation for Task 1 (Phase 1) of my FIT5226 Stage 2 Project. 

My agents (Type A and Type B) must act as independent learners using Tabular Q-Learning in a synchronized 5x5 grid-world. I am aiming for a High Distinction (HD). Generative AI is permitted, but I must perfectly understand and defend every line of code.

Here is my current codebase:
@configs, @src, @tests, @main

I need you to BRUTALLY AUDIT my code and HELP ME BUILD/REFINE the simulation architecture. Evaluate my implementation strictly against the following MARL first principles and project mechanics. If I fail any of these, rewrite the specific Python method to fix it, providing Jupyter-compatible code blocks.

### 1. SYNCHRONIZED SIMULATION PHYSICS (The Environment)
Check my Grid World class and step function for these absolute rules:
*   **Simultaneous Stepping:** The step function must accept a joint action (for robots Type A and type B) and execute them simultaneously.
*   **End-of-Step Collisions:** A collision ONLY occurs if Agent A and Agent B land on the exact same coordinate at the *end* of the time step. Swapping cells is NOT a collision. Same-type robots do not collide. Does my collision logic strictly evaluate post-transition coordinates?
*   **The Stochastic Lake:** The lake must flip between 'dry' and 'flooded' stochastically at every single time step with a fixed probability P (e.g., 0.5). It cannot be static. Does the environment handle this transition correctly?
*   **Episode Termination:** The episode must ONLY terminate when both Agent A and Agent B have successfully completed one full delivery cycle (pick-up at target, drop-off at start).
*   **Strict Rewards:** Rewards must be pure physical consequences: Step = -5, Wait = -3, Collision = -20, Type A in water = -20, Reached Target = 10, Delivered = 50. Are there any heuristic/distance-based tricks? If so, delete them.

### 2. THE "EXPECTED VALUE" TRAP (The Q-Learning Algorithm)
In Task 1, Sarah warns to "account for both possible states of the lake in the next state." Standard Q-learning will fail here due to high variance.
*   **Algorithm Check:** I must use Tabular Q-Learning. No Deep-Q (DQN) is allowed. 
*   **The Bellman Update:** Scrutinize my Q-table update method. Because the lake flips with probability P, the agent cannot just update based on the *observed* next state. My code MUST calculate the Expected Value of the next state over the probabilities of the lake being dry vs. flooded:
    E[Q_next] = P_hat(Dry) * max Q(S_next, Dry, a') + P_hat(Flooded) * max Q(S_next, Flooded, a')
*   **Agent Observability:** The state space for each agent must be strictly: `positions, has_sample, lake_flooded`. The agents CANNOT observe the other agent's location. If my state space includes the other agent's location, my independent learning model is entirely compromised. Fix it immediately.
*   **Shared Policies:** There must be exactly ONE Q-table for Type A and ONE for Type B.

### 3. EVALUATION METRICS & JUPYTER SYNCHRONIZATION
To prove the agents actually learned the "traffic light" behavior for an HD, my Jupyter notebook needs rigorous evaluation code.
*   **Greedy Evaluation:** Ensure I have a separate evaluation loop that pauses training (e.g., every 1000 steps), sets epsilon=0 (pure greedy), and runs test episodes. 
*   **Metrics:** Generate Python plotting code (matplotlib) to track: 
    1. Average Evaluation Return over time.
    2. Average Episode Length (which should drop and stabilize).
    3. Collision Rate & Water Damage Rate per episode (which must drop to 0).
*   **Policy Visualization:** Generate a Jupyter-compatible function to print out a visual grid of the final optimal policies (arrows for N, S, E, W, and 'Wait') for both Type A and Type B, under both conditions of the lake (Dry vs. Flooded). 

Give me a brutal, no-nonsense assessment. Output the corrected, fully synchronized Python code for the Environment, the Agents, and the Training/Evaluation loops suitable for a Jupyter Notebook.
