import importlib
import math

import tensorflow as tf
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions

import util


class A3CAgent(base_agent.BaseAgent):
    def __init__(self, is_training, m_size, s_size, network, session, summary_writer, name="A3C/A3CAgent"):
        super().__init__()
        self.name = name
        self.is_training = is_training
        self.summary = []

        self.session = session
        self.summary_writer = summary_writer

        # Dimensions
        # Make sure that screen resolution is equal to minimap resolution
        assert(m_size == s_size)
        self.m_size = m_size
        self.s_size = s_size
        self.i_size = len(actions.FUNCTIONS)

        self.epsilon = [0.3, 0.3]

        # Network
        self.network = self._init_network(network)
        self.net_output = {}

        # Tensor dictionaries
        self.features = {}
        self.valid_actions = {}
        self.targets = {}

        # Operations
        self.train_op = None
        self.summary_op = None

        # Saver
        self.saver = None

    def _make_worker(self):
        pass

    def _build_optimizer(self, loss):
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
        gradients = optimizer.compute_gradients(loss)
        clipped_gradients = []
        for g, v in gradients:
            if g is None:
                continue
            self.summary.append(tf.summary.histogram(v.op.name, v))
            self.summary.append(tf.summary.histogram(v.op.name + '/gradient', g))
            g = tf.clip_by_norm(g, 10.0)
            clipped_gradients.append([g, v])
        self.train_op = optimizer.apply_gradients(clipped_gradients)
        self.summary_op = tf.summary.merge(self.summary)

    def _compute_log_probability(self):
        # Sum over spatial policy * target
        spatial_action_prob = tf.reduce_sum(self.net_output["spatial"] * self.targets["spatial"], axis=1)
        # Take log of clipped result
        spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))

        # Sum over non spatial policy * target
        non_spatial_action_prob = tf.reduce_sum(self.net_output["non_spatial"] * self.targets["non_spatial"], axis=1)

        # Sum over non_spatial policy * non_spatial mask
        valid_non_spatial_action_prob = tf.reduce_sum(self.net_output["non_spatial"] * self.valid_actions["non_spatial"],
                                                      axis=1)

        # Clip valid non_spatial action probabilities
        valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)

        # normalize
        non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob

        # Take log of clipped non spatial action probability
        non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))

        # Log histogram for tensorboard
        self.summary.append(tf.summary.histogram("spatial_action_prob", spatial_action_prob))
        self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

        return {"spatial": spatial_action_log_prob, "non_spatial": non_spatial_action_log_prob}

    def _compute_advantage(self):
        # R - V(s)
        return tf.stop_gradient(self.targets["value"] - self.net_output["value"])

    def _compute_policy_loss(self, action_log_probs, advantage):
        # -log(Policy) * advantage - Beta*Entropy(policy)
        action_log_prob = self.valid_actions["spatial"] * action_log_probs["spatial"] + action_log_probs["non_spatial"]

        policy_loss = -tf.reduce_mean(action_log_prob * advantage)  # missing entropy
        self.summary.append(tf.summary.scalar('policy_loss', policy_loss))

        return policy_loss

    def _compute_value_loss(self, advantage):
        # Sum((R - V(s))^2
        value_loss = -tf.reduce_mean(self.net_output["value"] * advantage)
        self.summary.append(tf.summary.scalar('value_loss', value_loss))

        return value_loss

    def _init_network(self, network):
        network_module, network_name = network.rsplit(".", 1)
        network_cls = getattr(importlib.import_module(network_module), network_name)

        return network_cls(self.m_size, self.s_size, self.i_size)

    def _init_inputs(self):
        return {
            "minimap": tf.placeholder(tf.float32, [None, util.minimap_channel(), self.m_size, self.m_size],
                                      name='m_feats'),
            "screen": tf.placeholder(tf.float32, [None, util.screen_channel(), self.s_size, self.s_size],
                                     name='s_feats'),
            "info": tf.placeholder(tf.float32, [None, self.s_size * self.s_size], name='i_feats')
        }

    def _init_valid_actions(self):
        return {
            "spatial": tf.placeholder(tf.float32, [None], name='valid_spatial_action'),
            "non_spatial": tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='valid_non_spatial_action')
        }

    def _init_target_actions(self):
        return {
            "spatial": tf.placeholder(tf.float32, [None, self.s_size ** 2], name='spatial_action_selected'),
            "non_spatial": tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                          name='non_spatial_action_selected'),
            "value": tf.placeholder(tf.float32, [None], name='value_target')
        }

    def build_model(self, reuse, device):
        with tf.variable_scope(self.name) and tf.device(device):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse

            # Build the network
            self.features = self._init_inputs()
            self.network.init_inputs(self.features)
            self.net_output = self.network.build()

            # Initialize placeholders for masks and targets
            self.valid_actions = self._init_valid_actions()
            self.targets = self._init_target_actions()

            # Compute losses
            action_log_probs = self._compute_log_probability()
            advantage = self._compute_advantage()

            policy_loss = self._compute_policy_loss(action_log_probs, advantage)
            value_loss = self._compute_value_loss(advantage)

            # To do: entropy regularization
            loss = policy_loss + value_loss
            self.summary.append(tf.summary.scalar('loss', loss))

            # Build optimizer
            self._build_optimizer(loss)

            self.saver = tf.train.Saver(max_to_keep=100)

    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)

    def reset(self):
        self.episodes += 1
        self.epsilon = [0.3, 0.3]

    def step(self, obs):
        minimap = np.array(obs.observation['minimap'], dtype=np.float32)
        minimap = np.expand_dims(util.preprocess_minimap(minimap), axis=0)
        screen = np.array(obs.observation['screen'], dtype=np.float32)
        screen = np.expand_dims(util.preprocess_screen(screen), axis=0)

        info = np.zeros([1, self.s_size * self.s_size], dtype=np.float32)

        # let the agent know what it can do
        info[0, obs.observation['available_actions']] = 1

        count = 0
        # general player information, such as player_id, minerals, army count
        for feature in obs.observation['player']:
            info[0, len(actions.FUNCTIONS) + count] = feature
            count += 1

        tmp_counter = 0
        # single select information
        for feature in obs.observation['player']:
            info[0, len(actions.FUNCTIONS) + count] = feature
            if tmp_counter == 2 or tmp_counter == 3 or tmp_counter == 4:
                if feature > 0:
                    info[0, len(actions.FUNCTIONS) + count] = math.log(feature)
            count += 1
            tmp_counter += 1

        feed_dict = {self.features["minimap"]: minimap,
                     self.features["screen"]: screen,
                     self.features["info"]: info}
        non_spatial_action, spatial_action = self.session.run(
            [self.net_output["non_spatial"], self.net_output["spatial"]],
            feed_dict=feed_dict)

        # Select an action and a spatial target
        non_spatial_action = non_spatial_action.ravel()
        spatial_action = spatial_action.ravel()
        valid_actions = obs.observation['available_actions']
        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        target = np.argmax(spatial_action)
        target = [int(target // self.s_size), int(target % self.s_size)]

        # Epsilon greedy exploration
        if self.is_training and np.random.rand() < self.epsilon[0]:
            act_id = np.random.choice(valid_actions)
        if self.is_training and np.random.rand() < self.epsilon[1]:
            dy = np.random.randint(-4, 5)
            target[0] = int(max(0, min(self.s_size - 1, target[0] + dy)))
            dx = np.random.randint(-4, 5)
            target[1] = int(max(0, min(self.s_size - 1, target[1] + dx)))

        # Set act_id and act_args
        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([target[1], target[0]])
                pass
            else:
                act_args.append([0])

        return actions.FunctionCall(act_id, act_args)

    def update(self, replay_buffer, discount, learning_rate, counter):
        # Compute value of last observation
        obs = replay_buffer[-1][-1]

        # If last observation, set reward = 0
        reward = 0

        if not obs.last():
            minimap = np.array(obs.observation["minimap"], dtype=np.float32)
            minimap = np.expand_dims(util.preprocess_minimap(minimap), axis=0)
            screen = np.array(obs.obsevation["screen"], dtype=np.float32)
            screen = np.expand_dims(util.preprocess_screen(screen), axis=0)
            info = np.zeros([1, self.s_size * self.s_size], dtype=np.float32)

            # let the agent know what it can do
            info[0, obs.observation['available_actions']] = 1

            count = 0
            # General player information, such as player_id, minerals, army count
            for feature in obs.observation["player"]:
                info[0, len(actions.FUNCTIONS) + count] = feature
                count += 1

            tmp_counter = 0
            # single select information
            for feature in obs.observation['player']:
                info[0, len(actions.FUNCTIONS) + count] = feature
                if tmp_counter == 2 or tmp_counter == 3 or tmp_counter == 4:
                    if feature > 0:
                        info[0, len(actions.FUNCTIONS) + count] = math.log(feature)

                count += 1
                tmp_counter += 1

            feed_dict = {self.features["minimap"]: minimap,
                         self.features["screen"]: screen,
                         self.features["info"]: info}

            reward = self.session.run(self.net_output["value"], feed_dict=feed_dict)[0]

        # Compute targets and masks
        value_target = np.zeros([len(replay_buffer)], dtype=np.float32)
        value_target[-1] = reward

        valid_spatial_action = np.zeros([len(replay_buffer)], dtype=np.float32)
        spatial_action_selected = np.zeros([len(replay_buffer), self.s_size ** 2], dtype=np.float32)
        valid_non_spatial_action = np.zeros([len(replay_buffer), len(actions.FUNCTIONS)], dtype=np.float32)
        non_spatial_action_selected = np.zeros([len(replay_buffer), len(actions.FUNCTIONS)], dtype=np.float32)

        minimaps = []
        screens = []
        infos = []

        replay_buffer.reverse()
        for i, [action, obs] in enumerate(replay_buffer):
            minimap = np.array(obs.observation["minimap"], dtype=np.float32)
            minimap = np.expand_dims(util.preprocess_minimap(minimap), axis=0)
            screen = np.array(obs.observation["screen"], dtype=np.float32)
            screen = np.expand_dims(util.preprocess_screen(screen), axis=0)
            info = np.zeros([1, self.s_size * self.s_size], dtype=np.float32)

            # let the agent know what it can do
            info[0, obs.observation['available_actions']] = 1

            count = 0
            # general player information, such as player_id, minerals, army count
            for feature in obs.observation['player']:
                info[0, len(actions.FUNCTIONS) + count] = feature
                count += 1

            tmp_counter = 0
            # single select information
            for feature in obs.observation['player']:
                info[0, len(actions.FUNCTIONS) + count] = feature
                if tmp_counter == 2 or tmp_counter == 3 or tmp_counter == 4:
                    if feature > 0:
                        info[0, len(actions.FUNCTIONS) + count] = math.log(feature)
                count += 1
                tmp_counter += 1

            minimaps.append(minimap)
            screens.append(screen)
            infos.append(info)

            reward = int(obs.observation["score_cumulative"][0])
            act_id = action.function
            act_args = action.arguments

            value_target[i] = reward + discount * value_target[i - 1]

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
        feed = {self.features["minimap"]: minimaps,
                self.features["screen"]: screens,
                self.features["info"]: infos,
                self.targets["value"]: value_target,
                self.valid_actions["spatial"]: valid_spatial_action,
                self.targets["spatial"]: spatial_action_selected,
                self.valid_actions["non_spatial"]: valid_non_spatial_action,
                self.targets["non_spatial"]: non_spatial_action_selected,
                self.learning_rate: learning_rate}

        _, summary = self.session.run([self.train_op, self.summary_op], feed_dict=feed)
        self.summary_writer.add_graph(self.session.graph)
        self.summary_writer.add_summary(summary, counter)

    def save_model(self, path, counter):
        self.saver.save(self.session, path + '/model.pkl', counter)

    def load_model(self, path):
        checkpoint = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        return int(checkpoint.model_checkpoint_path.split('-')[-1])
