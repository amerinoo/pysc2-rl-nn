import tensorflow as tf
import numpy as np
import threading

from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions

import util

from agents.a3c.estimators import PolicyEstimator, ValueEstimator
from runner import Runner

from absl import flags

FLAGS = flags.FLAGS
SNAPSHOT_PATH = FLAGS.snapshot_path + FLAGS.map + '/' + FLAGS.agent


global UPDATE_LOCK
UPDATE_LOCK = threading.Lock()


class Worker(BaseAgent):
    def __init__(self,
                 name,
                 env_fn,
                 device,
                 session,
                 is_training,
                 m_size,
                 s_size,
                 policy_net,
                 value_net,
                 network,
                 global_step_counter,
                 global_episode_counter,
                 map_name,
                 discount_factor=0.99,
                 summary_writer=None,
                 max_global_steps=0,
                 max_global_episodes=0):

        super().__init__()
        self.name = name
        self.env_fn = env_fn
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.max_global_episodes = max_global_episodes
        self.global_step = tf.train.get_global_step()
        self.global_step_counter = global_step_counter
        self.global_episode_counter = global_episode_counter

        self.device = device
        self.session = session
        self.is_training = is_training

        self.summary = []

        self.summary_writer = summary_writer
        self.map_name = map_name

        # Dimensions
        self.m_size = m_size
        self.s_size = s_size

        # Network
        self.global_policy_net = policy_net
        self.global_value_net = value_net

        # Tensor dictionaries
        self.valid_actions = {}

        # Saver
        self.saver = None

        # Placeholders
        self.features = util.init_feature_placeholders(m_size, s_size)

        # build the local model
        self._build_model(network, device)

    def _build_model(self, network, device):
        with tf.device(device):
            with tf.variable_scope(self.name):
                # Obtain state representation and fullyConv from shared network
                state, fc = network.build(self.features, summarize=self.summary_writer is not None)

                # Build the rest of the network
                self.policy_net = PolicyEstimator(
                    self.s_size, state, fc, summarize=self.summary_writer is not None)
                self.value_net = ValueEstimator(
                    fc, summarize=self.summary_writer is not None)

                # Create op: Copy global variables
                self.copy_params_op = util.make_copy_params_op(
                    tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
                    tf.contrib.slim.get_variables(scope=self.name+'/', collection=tf.GraphKeys.TRAINABLE_VARIABLES)
                )

                # Create op: Update global variables with local losses
                self.vnet_train_op = util.make_train_op(self.value_net, self.global_value_net)
                self.pnet_train_op = util.make_train_op(self.policy_net, self.global_policy_net)

                self.saver = tf.train.Saver(max_to_keep=100)

    def _policy_net_predict(self, obs):
        feed_dict = {
             self.features["minimap"]: util.minimap_obs(obs),
             self.features["screen"]: util.screen_obs(obs),
             self.features["info"]: util.info_obs(obs)
        }

        # Get spatial/non_spatial policies
        policy = self.session.run(
            {
                "spatial": self.policy_net.prediction["spatial"],
                "non_spatial": self.policy_net.prediction["non_spatial"]
            },
            feed_dict=feed_dict
        )

        policy["non_spatial"] = policy["non_spatial"].ravel()
        policy["spatial"] = policy["spatial"].ravel()

        return policy

    def _value_net_predict(self, obs):
        feed_dict = {
             self.features["minimap"]: util.minimap_obs(obs),
             self.features["screen"]: util.screen_obs(obs),
             self.features["info"]: util.info_obs(obs)
        }

        return self.session.run(self.value_net.prediction, feed_dict=feed_dict)

    def reset(self):
        self.episodes += 1
        self.steps = 0

    def run(self, coord, max_update_steps, max_local_steps):
        tf.logging.info("Running agent {}".format(self.name))
        env = self.env_fn()
        runner = Runner(self, env, max_update_steps, max_local_steps, self.max_global_steps)

        with self.session.as_default(), self.session.graph.as_default():
            is_session_done = False
            global_episode = 0

            try:
                while not coord.should_stop():
                    # Copy parameters from the global networks
                    self.session.run(self.copy_params_op)

                    # Collect some experience
                    for replay_buffer, is_done, global_step in runner.run_n_steps(self.global_step_counter):
                        if self.is_training:
                            with UPDATE_LOCK:
                                tf.logging.info("{}: UPDATE_LOCK acquired. Updating...".format(self.name))
                                self.update(replay_buffer, FLAGS.learning_rate, global_episode)
                                tf.logging.info("{}: UPDATE_LOCK released!".format(self.name))

                            if is_done:
                                global_episode = next(self.global_episode_counter)
                                tf.logging.info("{} - Global episode: {}".format(self.name, global_episode))

                                if FLAGS.save_replay:
                                    env.save_replay(self.name)
                            #     if counter % FLAGS.snapshot_step == 1:
                            #         agent.save_model(SNAPSHOT_PATH, counter)

                            # If global max steps reached or global max episodes reached, terminate session
                            if self.max_global_steps != 0 and global_step >= self.max_global_steps:
                                tf.logging.info("Reached max global step {}".format(global_step))
                                is_session_done = True
                            elif self.max_global_episodes != 0 and global_episode >= self.max_global_episodes:
                                tf.logging.info("Reached max global episodes {}".format(global_episode))
                                is_session_done = True
                            if is_session_done:
                                coord.request_stop()
                                return

                        elif is_done:
                            obs = replay_buffer[-1].observation
                            score = obs["score_cumulative"][0]
                            tf.logging.info("Your score is {}!".format(str(score)))

            except tf.errors.CancelledError:
                return

    def _exploit_random(self, valid_actions):
        # Choose a random valid action
        act_id = np.random.choice(valid_actions)

        # Choose random target
        target = [np.random.randint(0, self.s_size),
                  np.random.randint(0, self.s_size)]

        return act_id, target

    def _exploit_max(self, policy, valid_actions):
        # Choose 'best' valid action
        act_id = valid_actions[np.argmax(policy["non_spatial"][valid_actions])]
        target = np.argmax(policy["spatial"])

        # Resize to provided resolution
        # Example:
        #   target = 535 -> 535 // 64 = 8, 535 % 64 = 24
        #   target = [8, 24]
        target = [int(target // self.s_size), int(target % self.s_size)]

        return act_id, target

    def _exploit_distribution(self, policy, valid_actions):
        # Mask actions
        non_spatial_policy = policy["non_spatial"][valid_actions]

        # Normalize probabilities
        non_spatial_probs = non_spatial_policy/np.sum(non_spatial_policy)

        # Choose from normalized distribution
        act_id = np.random.choice(valid_actions, p=non_spatial_probs)
        target = np.random.choice(np.arange(len(policy["spatial"])), p=policy["spatial"])

        # Resize to provided resolution
        target = [int(target // self.s_size), int(target % self.s_size)]

        return act_id, target

    def step(self, obs):
        policy = self._policy_net_predict(obs)
        valid_actions = obs.observation['available_actions']
        # if np.random.random() < 0.3:
        #     act_id, target = self._exploit_random(valid_actions)
        # else:
        #     act_id, target = self._exploit_max(policy, valid_actions)
        act_id, target = self._exploit_distribution(policy, valid_actions)
        act_args = util.get_action_arguments(act_id, target)

        self.steps += 1

        return actions.FunctionCall(act_id, act_args)

    def update(self, replay_buffer, learning_rate, counter):
        # Compute value of last observation
        obs = replay_buffer[-1][-1]

        # If obs wasn't done, bootstrap from last state
        reward = 0.0
        if not obs.last():
            reward = self._value_net_predict(obs)

        # Preallocate array sizes for _*speed*_
        value_targets = np.zeros([len(replay_buffer)], dtype=np.float32)
        policy_targets = np.zeros([len(replay_buffer)], dtype=np.float32)
        value_targets[-1] = reward

        valid_spatial_actions = np.zeros([len(replay_buffer)], dtype=np.int32)
        # spatial_actions_selected = np.zeros([len(replay_buffer), self.s_size ** 2], dtype=np.int32)
        spatial_actions_selected = np.zeros(len(replay_buffer), dtype=np.int32)
        valid_non_spatial_actions = np.zeros([len(replay_buffer), len(actions.FUNCTIONS)], dtype=np.int32)
        # non_spatial_actions_selected = np.zeros([len(replay_buffer), len(actions.FUNCTIONS)], dtype=np.int32)
        non_spatial_actions_selected = np.zeros(len(replay_buffer), dtype=np.int32)

        # TODO: Preallocate sizes
        minimaps = []
        screens = []
        infos = []

        # Accumulate batch updates
        replay_buffer.reverse()
        for i, [action, obs] in enumerate(replay_buffer):
            # Update state
            minimaps.append(util.minimap_obs(obs))
            screens.append(util.screen_obs(obs))
            infos.append(util.info_obs(obs))

            # Get selected action
            act_id = action.function
            act_args = action.arguments

            # Update reward
            reward = int(obs.observation["score_cumulative"][0]) + self.discount_factor * reward
            policy_target = (reward - self._value_net_predict(obs))

            policy_targets[i] = policy_target   # Append advantage
            value_targets[i] = reward           # Append discounted reward

            # Append selected action
            valid_actions = obs.observation["available_actions"]
            valid_non_spatial_actions[i, valid_actions] = 1
            # non_spatial_actions_selected[i, act_id] = 1
            non_spatial_actions_selected[i] = act_id

            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * self.s_size + act_arg[0]
                    valid_spatial_actions[i] = 1
                    spatial_actions_selected[i] = ind
                    # spatial_actions_selected[i, ind] = 1

        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        infos = np.concatenate(infos, axis=0)

        # Train
        feed_dict = {
            self.features["minimap"]: minimaps,
            self.features["screen"]: screens,
            self.features["info"]: infos,
            self.policy_net.actions["spatial"]: spatial_actions_selected,
            self.policy_net.actions["non_spatial"]: non_spatial_actions_selected,
            self.policy_net.targets: policy_targets,
            self.policy_net.valid_actions["spatial"]: valid_spatial_actions,
            self.policy_net.valid_actions["non_spatial"]: valid_non_spatial_actions,
            # self.policy_net.learning_rate: learning_rate,
            self.features["minimap"]: minimaps,
            self.features["screen"]: screens,
            self.features["info"]: infos,
            self.value_net.targets: value_targets
            # self.value_net.learning_rate: learning_rate
        }

        fetch = [
            self.global_step,
            self.policy_net.loss,
            self.value_net.loss,
            self.pnet_train_op,
            self.vnet_train_op
        ]

        if self.summary_writer is None:
            global_step, pnet_loss, vnet_loss, _, _, = self.session.run(
                fetch,
                feed_dict
            )
        else:
            fetch.append(self.policy_net.summaries)
            fetch.append(self.value_net.summaries)

            global_step, pnet_loss, vnet_loss, _, _, pnet_summaries, vnet_summaries = self.session.run(
                fetch,
                feed_dict
            )

            # Write summaries
            self.summary_writer.add_graph(self.session.graph)
            self.summary_writer.add_summary(pnet_summaries, global_step)
            self.summary_writer.add_summary(vnet_summaries, global_step)

            self.summary_writer.flush()

        return pnet_loss, vnet_loss

    def save_model(self, path, counter):
        self.saver.save(self.session, path + '/model.pkl', counter)

    def load_model(self, path):
        checkpoint = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        return int(checkpoint.model_checkpoint_path.split('-')[-1])
