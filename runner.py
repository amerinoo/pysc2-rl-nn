import time


class Runner(object):
    def __init__(self, agent, env, update_period=0, max_local_steps=0, max_global_steps=0):
        self.agent = agent
        self.env = env
        self.update_period = update_period
        self.max_local_steps = max_local_steps
        self.max_global_steps = max_global_steps

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

    def run_n_steps(self, global_step_counter):
        """A run loop to have agents and an environment interact."""
        if self.is_first_step:
            self.start_time = time.time()

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

                global_step = next(global_step_counter)

                # Yield current buffer if update_period is reached or episode finished
                is_update_ready = self.agent.steps % self.update_period == 0
                is_done = self.timestep.last() \
                    or (self.max_local_steps != 0 and self.agent.steps >= self.max_local_steps) \
                    or (self.max_global_steps != 0 and global_step >= self.max_global_steps)

                replay_buffer.append([action, self.timestep])
                if is_update_ready or is_done:
                    print("Updating {} at step {}/{}".format(self.agent.name, self.agent.steps, self.max_local_steps))
                    yield replay_buffer, is_done, global_step
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