import time


class Runner(object):
    def __init__(self, agent, env, update_period=0):
        self.agent = agent
        self.env = env
        self.update_period = update_period

        self.timestep = None
        self.start_time = None

        self.is_done = False
        self.is_first_step = True

        action_spec = self.env.action_spec()
        observation_spec = self.env.observation_spec()
        self.agent.setup(observation_spec, action_spec)

    def _reset(self):
        self.is_first_step = True
        self.timestep = None

    def run_n_steps(self):
        """A run loop to have agents and an environment interact."""
        if self.is_first_step:
            self.start_time = time.time()

        steps = 0
        replay_buffer = []

        try:
            # If starting new episode, reset agent and environment
            if self.is_first_step:
                timesteps = self.env.reset()
                self.timestep = timesteps[0]
                self.agent.reset()

            while True:
                assert(self.timestep is not None)

                action = self.agent.step(self.timestep)

                steps += 1

                # Yield current buffer if update_period is reached or episode finished
                is_update_ready = steps % self.update_period == 0
                is_done = self.timestep.last()

                replay_buffer.append([action, self.timestep])
                if is_update_ready or is_done:
                    print("Updating at {} - {}".format(steps, self.update_period))
                    yield replay_buffer, is_done
                if is_done:
                    self._reset()
                    break

                timesteps = self.env.step([action])
                self.timestep = timesteps[0]

        except KeyboardInterrupt:
            pass
        finally:
            elapsed_time = time.time() - self.start_time
            print("Took %.3f seconds for %s steps: %.3f steps/second"
                  % (elapsed_time, self.agent.steps, self.agent.steps/elapsed_time))

    # def run_loop(self, agent, env, num_episodes=0, max_agent_steps=0):
    #     """A run loop to have agents and an environment interact."""
    #     start_time = time.time()
    #
    #     action_spec = env.action_spec()
    #     observation_spec = env.observation_spec()
    #     agent.setup(observation_spec, action_spec)
    #
    #     total_episodes = 0
    #
    #     try:
    #         while True:
    #             timesteps = env.reset()
    #             agent.reset()
    #
    #             if num_episodes != 0 and total_episodes >= num_episodes:
    #                 print("Reached max episodes")
    #                 break
    #
    #             while True:
    #                 timestep = timesteps[0]
    #                 action = agent.step(timestep)
    #
    #                 is_done = (max_agent_steps != 0 and agent.steps >= max_agent_steps) or timestep.last()
    #                 yield [action, timestep], is_done
    #                 if is_done:
    #                     break
    #
    #                 timesteps = env.step([action])
    #
    #             total_episodes += 1
    #     except KeyboardInterrupt:
    #         pass
    #     finally:
    #         elapsed_time = time.time() - start_time
    #         print("Took %.3f seconds for %s steps: %.3f steps/second"
    #               % (elapsed_time, agent.steps, agent.steps / elapsed_time))