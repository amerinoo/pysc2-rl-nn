import time


def run_loop(agent, env, num_episodes=0, max_agent_steps=0):
    """A run loop to have agents and an environment interact."""
    start_time = time.time()

    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    agent.setup(observation_spec, action_spec)

    total_episodes = 0

    try:
        while True:
            timesteps = env.reset()
            agent.reset()

            if total_episodes >= num_episodes:
                print("Reached max episodes")
                break

            while True:
                timestep = timesteps[0]
                action = agent.step(timestep)

                is_done = (agent.steps >= max_agent_steps) or timestep.last()
                yield [action, timestep], is_done
                if is_done:
                    break

                timesteps = env.step([action])

            total_episodes += 1
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f steps/second"
              % (elapsed_time, agent.steps, agent.steps/elapsed_time))
