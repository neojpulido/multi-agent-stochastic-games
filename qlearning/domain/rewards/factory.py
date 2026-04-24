def get_reward_provider(config):
    if not config.is_multi_agent:
        return TransportReward(config.env)
    else:
        # Phase 1 = Sarah, Phase 2 = Robert
        return CoordinationReward(config.env, phase=config.phase)