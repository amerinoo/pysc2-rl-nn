import logging

import tensorflow as tf
import numpy as np

from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions

import util

from runner import Runner

from absl import flags

logging.basicConfig(level=logging.INFO)

FLAGS = flags.FLAGS
SNAPSHOT_PATH = FLAGS.snapshot_path + FLAGS.map + '/' + FLAGS.agent


class Worker(BaseAgent):
    def __init__(self,
                 name,
                 env_fn,
                 reuse,
                 device,
                 session,
                 is_training,
                 m_size,
                 s_size,
                 global_net,
                 local_net,
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
        self.global_net = global_net
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
        self.network = util.init_network(local_net, self.m_size, self.s_size, len(actions.FUNCTIONS))
        self.net_output = {}

        # self.epsilon = 0.05

        # Tensor dictionaries
        self.features = {}
        self.valid_actions = {}
        self.targets = {}

        # Operations
        self.train_op = None
        self.summary_op = None

        # Saver
        self.saver = None

        # build the local model
        self._build_model(reuse, device)

    def _build_model(self, reuse, device):
        with tf.variable_scope("shared") and tf.device(device):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse

            # Build the network
            self.features = self._init_input_placeholders()
            self.network.init_inputs(self.features)
            self.net_output = self.network.build()

            # Initialize placeholders for masks and targets
            self.valid_actions = self._init_valid_action_placeholders()
            self.targets = self._init_target_placeholders()

            # Compute losses
            action_log_probs = self._compute_log_probability()
            advantage = self._compute_advantage()
            weighted_entropy = self._compute_weighted_entropy(action_log_probs, B=0.01)

            policy_loss = self._compute_policy_loss(action_log_probs, advantage)
            value_loss = self._compute_value_loss(advantage)

            loss = policy_loss + value_loss + weighted_entropy
            if self.summary_writer is not None:
                self.summary.append(tf.summary.scalar('loss', loss))

            # Build optimizer
            self._build_optimizer(loss)

            self.saver = tf.train.Saver(max_to_keep=100)

    def _build_optimizer(self, loss):
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
        gradients = optimizer.compute_gradients(loss)
        clipped_gradients = []
        for g, v in gradients:
            if g is None:
                continue

            if self.summary_writer is not None:
                self.summary.append(tf.summary.histogram(v.op.name, v))
                self.summary.append(tf.summary.histogram(v.op.name + '/gradient', g))
            g = tf.clip_by_norm(g, 10.0)
            clipped_gradients.append([g, v])
        self.train_op = optimizer.apply_gradients(clipped_gradients)

        if self.summary_writer is not None:
            self.summary_op = tf.summary.merge(self.summary)

    def _compute_weighted_entropy(self, action_log_probs, B):
        spatial_entropy = -tf.reduce_sum(tf.reduce_sum(self.net_output["spatial"], axis=1) * action_log_probs["spatial"])
        non_spatial_entropy = -tf.reduce_sum(tf.reduce_sum(self.net_output["non_spatial"], axis=1) * action_log_probs["non_spatial"])
        return B * (spatial_entropy + non_spatial_entropy)

    def _compute_log_probability(self):
        # Log of Sum over spatial policy * target
        spatial_action_prob = tf.reduce_sum(self.net_output["spatial"] * self.targets["spatial"], axis=1)
        spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))

        # Sum over non spatial policy * target
        non_spatial_action_prob = tf.reduce_sum(self.net_output["non_spatial"] * self.targets["non_spatial"], axis=1)

        # Sum over non_spatial policy * non_spatial mask
        valid_non_spatial_action_prob = tf.reduce_sum(
            self.net_output["non_spatial"] * self.valid_actions["non_spatial"],
            axis=1)

        # Clip valid non_spatial action probabilities
        valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)

        non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob

        # Take log of clipped non spatial action probability
        non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))

        # Log histogram for tensorboard
        if self.summary_writer is not None:
            self.summary.append(tf.summary.histogram("spatial_action_prob", spatial_action_prob))
            self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

        return {"spatial": spatial_action_log_prob, "non_spatial": non_spatial_action_log_prob}

    def _compute_advantage(self):
        # R - V(s)
        return tf.stop_gradient(self.targets["value"] - self.net_output["value"])

    def _compute_policy_loss(self, action_log_probs, advantage):
        # -log(Policy) * advantage - B*Entropy(policy)
        action_log_prob = self.valid_actions["spatial"] * action_log_probs["spatial"] + action_log_probs["non_spatial"]

        policy_loss = -tf.reduce_mean(action_log_prob * advantage)
        if self.summary_writer is not None:
            self.summary.append(tf.summary.scalar('policy_loss', policy_loss))

        return policy_loss

    def _compute_value_loss(self, advantage):
        # Sum((R - V(s))^2
        # TODO: Compare differences between implementations
        value_loss = -tf.reduce_mean(self.net_output["value"] * advantage)
        if self.summary_writer is not None:
            self.summary.append(tf.summary.scalar('value_loss', value_loss))

        return value_loss

    def _init_input_placeholders(self):
        return {
            "minimap": tf.placeholder(tf.float32, [None, util.minimap_channel_size(), self.m_size, self.m_size],
                                      name='m_feats'),
            "screen": tf.placeholder(tf.float32, [None, util.screen_channel_size(), self.s_size, self.s_size],
                                     name='s_feats'),
            "info": tf.placeholder(tf.float32, [None, util.structured_channel_size()], name='i_feats')
        }

    def _init_target_placeholders(self):
        return {
            "spatial": tf.placeholder(tf.float32, [None, self.s_size ** 2], name='spatial_action_selected'),
            "non_spatial": tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                          name='non_spatial_action_selected'),
            "value": tf.placeholder(tf.float32, [None], name='value_target')
        }

    def _init_valid_action_placeholders(self):
        return {
            "spatial": tf.placeholder(tf.float32, [None], name='valid_spatial_action'),
            "non_spatial": tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='valid_non_spatial_action')
        }

    def _value_net_predict(self, obs):
        minimap = util.minimap_obs(obs)
        screen = util.screen_obs(obs)
        info = util.info_obs(obs)

        feed_dict = {self.features["minimap"]: minimap,
                     self.features["screen"]: screen,
                     self.features["info"]: info}

        return self.session.run(self.net_output["value"], feed_dict=feed_dict)[0]

    def reset(self):
        self.episodes += 1
        self.steps = 0

    def run(self, coord, max_update_steps, max_local_steps):
        env = self.env_fn()
        runner = Runner(self, env, max_update_steps, max_local_steps, self.max_global_steps)

        with self.session.as_default(), self.session.graph.as_default():
            is_session_done = False
            global_episode = 0

            try:
                while not coord.should_stop():
                    # Collect some experience
                    for replay_buffer, is_done, global_step in runner.run_n_steps(self.global_step_counter):
                        if self.is_training:
                            self.update(replay_buffer, FLAGS.learning_rate, global_episode)

                            if is_done:
                                global_episode = next(self.global_episode_counter)
                                logging.info("{} - Global episode: {}".format(self.name, global_episode))

                                if FLAGS.save_replay:
                                    env.save_replay(self.name)
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

            except tf.errors.CancelledError:
                return

    def _explore(self, valid_actions):
        # Choose a random action
        act_id = np.random.choice(valid_actions)

        # Choose random target
        target = [np.random.randint(0, self.s_size),
                  np.random.randint(0, self.s_size)]

        return act_id, target

    def _exploit(self, valid_actions, spatial_action, non_spatial_action):
        # Choose 'best' action
        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        target = np.argmax(spatial_action)

        # Resize to provided resolution
        target = [int(target // self.s_size), int(target % self.s_size)]

        return act_id, target

    def _get_action_arguments(self, act_id, target):
        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([target[1], target[0]])
                pass
            else:
                act_args.append([0])

        return act_args

    def step(self, obs):
        minimap = util.minimap_obs(obs)
        screen = util.screen_obs(obs)
        info = util.info_obs(obs)  # Structured data not obtained from pixels

        feed_dict = {self.features["minimap"]: minimap,
                     self.features["screen"]: screen,
                     self.features["info"]: info}

        # Get spatial/non_spatial policies
        non_spatial_action, spatial_action = \
            self.session.run([self.net_output["non_spatial"], self.net_output["spatial"]], feed_dict=feed_dict)

        non_spatial_action = non_spatial_action.ravel()
        spatial_action = spatial_action.ravel()
        valid_actions = obs.observation['available_actions']

        # # Epsilon greedy exploration
        # if self.is_training and np.random.rand() < self.epsilon:
        #     # Explore random action/target
        #     act_id, target = self._explore(valid_actions)
        # else:
        # Exploit the 'best' policy
        act_id, target = self._exploit(valid_actions, spatial_action, non_spatial_action)

        act_args = self._get_action_arguments(act_id, target)

        self.steps += 1
        if self.name[-1] == '0':
            logging.info("{} - Action at step {}: {}({})"
                          .format(self.name, self.steps, actions.FUNCTIONS[act_id], act_args))
        return actions.FunctionCall(act_id, act_args)

    def update(self, replay_buffer, learning_rate, counter):
        # Compute value of last observation
        obs = replay_buffer[-1][-1]

        # If last observation, set reward = 0
        reward = 0.0
        if not obs.last():
            # Otherwise bootstrap from last state
            reward = self._value_net_predict(obs)

        # Preallocate arrays sizes for _*speed*_
        value_targets = np.zeros([len(replay_buffer)], dtype=np.float32)
        value_targets[-1] = reward

        valid_spatial_action = np.zeros([len(replay_buffer)], dtype=np.float32)
        spatial_action_selected = np.zeros([len(replay_buffer), self.s_size ** 2], dtype=np.float32)
        valid_non_spatial_action = np.zeros([len(replay_buffer), len(actions.FUNCTIONS)], dtype=np.float32)
        non_spatial_action_selected = np.zeros([len(replay_buffer), len(actions.FUNCTIONS)], dtype=np.float32)

        # Accumulate batch updates
        minimaps = []
        screens = []
        infos = []

        replay_buffer.reverse()
        for i, [action, obs] in enumerate(replay_buffer):
            minimap = util.minimap_obs(obs)
            screen = util.screen_obs(obs)
            info = util.info_obs(obs)

            minimaps.append(minimap)
            screens.append(screen)
            infos.append(info)

            reward = int(obs.observation["score_cumulative"][0])
            act_id = action.function
            act_args = action.arguments

            value_targets[i] = reward + self.discount_factor * value_targets[i - 1]

            valid_actions = obs.observation["available_actions"]
            valid_non_spatial_action[i, valid_actions] = 1
            non_spatial_action_selected[i, act_id] = 1

            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * self.s_size + act_arg[0]
                    valid_spatial_action[i] = 1
                    spatial_action_selected[i, ind] = 1

        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        infos = np.concatenate(infos, axis=0)

        # Train
        feed_dict = {self.features["minimap"]: minimaps,
                     self.features["screen"]: screens,
                     self.features["info"]: infos,
                     self.targets["value"]: value_targets,
                     self.valid_actions["spatial"]: valid_spatial_action,
                     self.targets["spatial"]: spatial_action_selected,
                     self.valid_actions["non_spatial"]: valid_non_spatial_action,
                     self.targets["non_spatial"]: non_spatial_action_selected,
                     self.learning_rate: learning_rate}

        if self.summary_writer is not None:
            _, summary = self.session.run([self.train_op, self.summary_op], feed_dict=feed_dict)
            self.summary_writer.add_graph(self.session.graph)
            self.summary_writer.add_summary(summary, counter)
        else:
            self.session.run(self.train_op, feed_dict=feed_dict)

    def save_model(self, path, counter):
        self.saver.save(self.session, path + '/model.pkl', counter)

    def load_model(self, path):
        checkpoint = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        return int(checkpoint.model_checkpoint_path.split('-')[-1])
