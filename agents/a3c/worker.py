import importlib
import itertools
import math

import tensorflow as tf
import numpy as np

from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions

import util

from runner import Runner

from absl import flags

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
                 policy_net,
                 value_net,
                 core_net,
                 global_counter,
                 map_name,
                 discount_factor=0.99,
                 summary_writer=None,
                 max_global_steps=None):

        super().__init__()
        self.name = name
        self.env_fn = env_fn
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_step = tf.train.get_global_step()
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.local_counter = itertools.count()

        self.device = device
        self.session = session
        self.is_training = is_training

        self.summary = []

        self.summary_writer = summary_writer
        self.map_name = map_name

        # Dimensions
        self.m_size = m_size
        self.s_size = s_size
        self.i_size = len(actions.FUNCTIONS)

        # Network
        self.network = self._init_network(core_net)
        self.net_output = {}

        self.epsilon = [0.1, 0.3]

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

            # ToDo: entropy regularization
            loss = policy_loss + value_loss
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

    def _compute_log_probability(self):
        # Log of Sum over spatial policy * target
        spatial_action_prob = tf.reduce_sum(self.net_output["spatial"] * self.targets["spatial"], axis=1)
        spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))

        # Sum over non spatial policy * target
        non_spatial_action_prob = tf.reduce_sum(self.net_output["non_spatial"] * self.targets["non_spatial"], axis=1)

        # Sum over non_spatial policy * non_spatial mask
        valid_non_spatial_action_prob = tf.reduce_sum(self.net_output["non_spatial"] * self.valid_actions["non_spatial"],
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

        # TODO: Entropy regularization
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

    def _init_inputs(self):
        return {
            "minimap": tf.placeholder(tf.float32, [None, util.minimap_channel(), self.m_size, self.m_size],
                                      name='m_feats'),
            "screen": tf.placeholder(tf.float32, [None, util.screen_channel(), self.s_size, self.s_size],
                                     name='s_feats'),

            # TODO: This is likely incorrect
            "info": tf.placeholder(tf.float32, [None, self.s_size * self.s_size], name='i_feats')
        }

    def _init_network(self, network):
        network_module, network_name = network.rsplit(".", 1)
        network_cls = getattr(importlib.import_module(network_module), network_name)

        return network_cls(self.m_size, self.s_size, self.i_size)

    def _init_target_actions(self):
        return {
            "spatial": tf.placeholder(tf.float32, [None, self.s_size ** 2], name='spatial_action_selected'),
            "non_spatial": tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                          name='non_spatial_action_selected'),
            "value": tf.placeholder(tf.float32, [None], name='value_target')
        }

    def _init_valid_actions(self):
        return {
            "spatial": tf.placeholder(tf.float32, [None], name='valid_spatial_action'),
            "non_spatial": tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='valid_non_spatial_action')
        }

    def _value_net_predict(self, obs):
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

        return self.session.run(self.net_output["value"], feed_dict=feed_dict)[0]

    def reset(self):
        self.episodes += 1
        self.epsilon = [0.3, 0.3]

    def run(self, coord, max_update_steps):
        env = self.env_fn()
        runner = Runner(self, env, max_update_steps)

        with self.session.as_default(), self.session.graph.as_default():
            # counter = 0
            try:
                while not coord.should_stop():
                    # for recorder, is_done in self.run_n_steps(t_max):
                    #     pass
                    #replay_buffer = []
                    for recorder, is_done in runner.run_n_steps():
                        pass
                    #    if FLAGS.training:
                    #        replay_buffer.append(recorder)
                            # if is_done:
                            #     counter += 1
                            #     # Learning rate schedule
                            #     learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
                            #     agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
                            #     replay_buffer = []
                            #     if counter % FLAGS.snapshot_step == 1:
                            #         agent.save_model(SNAPSHOT_PATH, counter)
                            #     if counter >= FLAGS.num_episodes:
                            #         break
                        # elif is_done:
                        #     obs = recorder[-1].observation
                        #     score = obs["score_cumulative"][0]
                        #     print('Your score is ' + str(score) + '!')
                    if FLAGS.save_replay:
                        env.save_replay(self.name)
            except tf.errors.CancelledError:
                return

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
        reward = 0.0
        if not obs.last():
            reward = self._value_net_predict(obs)

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
        feed_dict = {self.features["minimap"]: minimaps,
                self.features["screen"]: screens,
                self.features["info"]: infos,
                self.targets["value"]: value_target,
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
