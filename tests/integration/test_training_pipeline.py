from qlearning import TransportGridWorld, TabularQAgent

def test_training_convergence(default_experiment_config):
    """Integration test to verify that the agent can converge on a small grid."""
    env = TransportGridWorld(config=default_experiment_config.env)
    agent = TabularQAgent(config=default_experiment_config.agent)

    # Train for a small amount of episodes
    for _ in range(2000):
        previous_state = env.reset()
        done = False
        while not done:
            action = agent.select_action(previous_state)
            current_state, reward, done = env.execute_step(action)
            agent.apply_learning_step(previous_state, action, reward, current_state, done)
            previous_state = current_state

    # Verify success on a few random resets
    for _ in range(5):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 10:
            action = agent.select_action(state, use_greedy=True)
            state, _, done = env.execute_step(action)
            steps += 1
        assert done, "Agent failed to solve the task after training"


