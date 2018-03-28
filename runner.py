import time
import threading

from tensorflow import logging, errors
from absl import flags

FLAGS = flags.FLAGS
# SNAPSHOT_PATH = FLAGS.snapshot_path + FLAGS.map + '/' + FLAGS.agent

global UPDATE_LOCK
UPDATE_LOCK = threading.Lock()


class Runner(object):
    def __init__(self,
                 agent,
                 env,
                 is_training,
                 global_step_counter,
                 global_episode_counter,
                 update_period=0,
                 max_local_steps=0,
                 max_global_steps=0,
                 max_global_episodes=0):

        self.agent = agent
        self.env = env
        self.is_training = is_training
        self.update_period = update_period
        self.global_step_counter = global_step_counter
        self.global_episode_counter = global_episode_counter
        self.max_local_steps = max_local_steps
        self.max_global_steps = max_global_steps
        self.max_global_episodes = max_global_episodes

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

    def _run_n_steps(self, global_step_counter):
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
                assert (self.timestep is not None)

                action = self.agent.step(self.timestep)
                global_step = next(global_step_counter)

                # Yield current buffer if update_period is reached or episode finished
                is_update_ready = self.update_period != 0 and self.agent.steps % self.update_period == 0
                is_done = self.timestep.last() \
                          or (self.max_local_steps != 0 and self.agent.steps >= self.max_local_steps) \
                          or (self.max_global_steps != 0 and global_step >= self.max_global_steps)

                replay_buffer.append([action, self.timestep])
                if is_update_ready or is_done:
                    logging.info(
                        "{}: step {}/{}".format(self.agent.name, self.agent.steps, self.max_local_steps))
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
            logging.info("Took %.3f seconds for %s steps: %.3f steps/second"
                         % (elapsed_time, self.agent.steps, self.agent.steps / elapsed_time))

    def run(self, coord):
        logging.info("Running agent {}".format(self.agent.name))
        with self.agent.session.as_default(), self.agent.session.graph.as_default():
            is_session_done = False
            global_episode = 0

            try:
                while not coord.should_stop():
                    # Copy parameters from the global networks
                    self.agent.session.run(self.agent.copy_params_op)

                    # Collect some experience
                    for replay_buffer, is_done, global_step in self._run_n_steps(self.global_step_counter):
                        if self.is_training:
                            # if not UPDATE_LOCK.locked():
                            with UPDATE_LOCK:
                                logging.info("{}: UPDATE_LOCK acquired. Updating...".format(self.agent.name))
                                self.agent.update(replay_buffer)
                            logging.info("{}: UPDATE_LOCK released!".format(self.agent.name))

                            if is_done:
                                global_episode = next(self.global_episode_counter)
                                logging.info(
                                    "{} - Finished global episode: {}".format(self.agent.name, global_episode))

                                if FLAGS.save_replay:
                                    self.env.save_replay(self.agent.name)
                            #     if counter % FLAGS.snapshot_step == 1:
                            #         agent.save_model(SNAPSHOT_PATH, counter)

                            # If global max steps reached or global max episodes reached, terminate session
                            if self.max_global_steps != 0 and global_step >= self.max_global_steps:
                                logging.info("Reached max global step {}".format(global_step))
                                is_session_done = True
                            elif self.max_global_episodes != 0 and global_episode >= self.max_global_episodes:
                                logging.info("Reached max global episodes {}".format(global_episode))
                                is_session_done = True

                            if is_session_done:
                                coord.request_stop()
                                return

                        elif is_done:
                            obs = replay_buffer[-1].observation
                            score = obs["score_cumulative"][0]
                            logging.info("Your score is {}!".format(str(score)))

            except errors.CancelledError:
                return
