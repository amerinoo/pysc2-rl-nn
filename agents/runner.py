import csv
import time
import threading

from tensorflow import logging, errors
from absl import flags

EPISODE_LOCK = threading.Lock()
UPDATE_LOCK = threading.Lock()
STEP_LOCK = threading.Lock()
SAVE_LOCK = threading.Lock()
FLAGS = flags.FLAGS


class Runner(object):
    def __init__(self,
                 config,
                 agent,
                 env,
                 global_step_counter,
                 global_episode_counter,
                 saver=None):

        self.agent = agent
        self.env = env
        self.config = config
        self.is_training = config.training
        self.update_period = config.max_update_steps
        self.global_step_counter = global_step_counter
        self.global_episode_counter = global_episode_counter
        self.max_local_steps = config.max_steps
        self.max_global_steps = config.max_global_steps
        self.max_global_episodes = config.num_episodes
        self.saver = saver

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
                with STEP_LOCK:
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
                    replay_buffer = []
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
                    # Copy parameters from the global network
                    self.agent.session.run(self.agent.copy_params_op)

                    # Collect some experience
                    for replay_buffer, is_done, global_step in self._run_n_steps(self.global_step_counter):
                        if self.is_training:
                            self.agent.update(replay_buffer)

                            if is_done:
                                with EPISODE_LOCK:
                                    global_episode = next(self.global_episode_counter)
                                    obs = replay_buffer[-1][1].observation
                                    score = obs["score_cumulative"][0]
                                    self.log_episode(global_episode, score)

                                logging.info(
                                    "{} - Finished global episode: {}".format(self.agent.name, global_episode))

                                if FLAGS.save_replay:
                                    self.env.save_replay(self.agent.name)

                            if self.saver is not None and \
                                    (global_episode % FLAGS.snapshot_step == 0
                                     or global_episode == 500
                                     or global_episode == 1000
                                     or global_episode == 1500
                                     or global_episode == 2000
                                     or global_episode == 2500
                                     or global_episode == 3000):
                                with SAVE_LOCK:
                                    self.saver.save(self.agent.session, self.config.snapshot_path + '/model.pkl', global_episode)

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

            except errors.CancelledError:
                return

    def log_episode(self, episode, reward):
        logging.info("Logging reward: %d", reward)
        with open('rewards.csv', 'a', newline='') as csvfile:
            w = csv.writer(csvfile, delimiter=',')
            w.writerow([episode, reward, self.config.learning_rate, self.config.beta, self.config.eta])
