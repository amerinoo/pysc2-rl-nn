import tensorflow as tf
import tensorflow.contrib.layers as layers
import util
import pysc2.lib.actions as actions

class PolicyEstimator(object):
    def __init__(self, m_size, s_size, network, reuse=False, trainable=True, summary_writer=None):
        # Make sure that screen resolution is equal to minimap resolution
        assert (m_size == s_size)
        self.m_size = m_size
        self.s_size = s_size

        self.network = network

        # Input placeholders
        self.features = {
            "minimap": tf.placeholder(
                tf.float32, [None, util.minimap_channel_size(), self.m_size, self.m_size],
                name='minimap_features'
            ),
            "screen": tf.placeholder(
                tf.float32, [None, util.screen_channel_size(), self.s_size, self.s_size],
                name='screen_features'
            ),
            "info": tf.placeholder(
                tf.float32, [None, util.structured_channel_size()],
                name='info_features'
            )
        }

        # Target placeholders
        self.targets = tf.placeholder(
            tf.float32, [None],
            name='value'
        )

        # Mask placeholders
        self.valid_actions = {
            "spatial": tf.placeholder(
                tf.float32, [None],
                name='valid_spatial_action'
            ),
            "non_spatial": tf.placeholder(
                tf.float32, [None, len(actions.FUNCTIONS)],
                name='valid_non_spatial_action'
            )
        }

        # Action placeholders
        self.actions = {
            "spatial": tf.placeholder(
                tf.int32, [None, s_size ** 2],
                name='spatial_action_selected'
            ),
            "non_spatial": tf.placeholder(
                tf.int32, [None, len(actions.FUNCTIONS)],
                name='non_spatial_action_selected'
            )
        }

        # Batch size = number of steps fed to network
        batch_size = tf.shape(self.targets)[0]

        with tf.variable_scope("shared", reuse=reuse):
            self.state, self.fc = self.network.build(self.features)

        with tf.variable_scope("policy_net"):
            self.prediction = {
                "spatial":
                    tf.nn.softmax(
                        layers.flatten(
                            layers.conv2d(
                                self.state,
                                num_outputs=1,
                                kernel_size=1,
                                stride=1,
                                activation_fn=None,
                                scope='spatial_probs'
                            )
                        )
                    ),
                "non_spatial":
                    layers.fully_connected(
                        self.fc,
                        num_outputs=len(actions.FUNCTIONS),
                        activation_fn=tf.nn.softmax,
                        scope='non_spatial_probs'
                    )
            }

            # H(π) = -Σ(π(s) * log(π(s))) : over batched states
            self.spatial_entropy = -tf.reduce_sum(
                self.prediction["spatial"] * tf.log(self.prediction["spatial"]), 1,
                name="spatial_entropy"
            )
            self.non_spatial_entropy = -tf.reduce_sum(
                self.prediction["non_spatial"] * tf.log(self.prediction["non_spatial"]), 1,
                name="non_spatial_entropy"
            )
            self.entropy_mean = tf.reduce_mean([
                tf.reduce_mean(self.spatial_entropy),
                tf.reduce_mean(self.non_spatial_entropy)
            ], name="entropy_mean")

            picked_spatial_actions = tf.argmax(self.actions["spatial"], axis=1)
            picked_non_spatial_actions = tf.argmax(self.actions["non_spatial"], axis=1)
            self.picked_non_spatial_probs = \
                tf.gather(tf.reshape(self.prediction["non_spatial"], [-1]), picked_non_spatial_actions)
            self.picked_spatial_probs = \
                tf.gather(tf.reshape(self.prediction["spatial"], [-1]), picked_spatial_actions)

            # Policy Loss: L = -log(π(s)) * A(s) - β*H(π) : over batched states
            self.spatial_losses = \
                -(tf.log(self.picked_spatial_probs) * self.targets) + 0.01 * self.spatial_entropy
            self.non_spatial_losses = \
                -(tf.log(self.picked_non_spatial_probs) * self.targets) + 0.01 * self.non_spatial_entropy
            self.loss = tf.reduce_sum([
                tf.reduce_sum(self.spatial_losses),
                tf.reduce_sum(self.non_spatial_losses)
            ], name="policy_loss")

            if trainable:
                self.learning_rate = 1e-6  # tf.placeholder(tf.float32, None, name="learning_rate")
                self.optimizer = tf.train.RMSPropOptimizer(
                    self.learning_rate, decay=0.99, epsilon=1e-6,
                    name="policy_optimizer"
                )
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[g, v] for g, v in self.grads_and_vars if g is not None]
                self.train_op = self.optimizer.apply_gradients(
                    self.grads_and_vars,
                    global_step=tf.train.get_global_step(),
                    name="policy_grads"
                )


class ValueEstimator(object):
    def __init__(self, m_size, s_size, network, reuse=False, trainable=True, summary_writer=None):
        # Make sure that screen resolution is equal to minimap resolution
        assert (m_size == s_size)
        self.m_size = m_size
        self.s_size = s_size

        self.network = network

        # Input placeholders
        self.features = {
            "minimap": tf.placeholder(
                tf.float32, [None, util.minimap_channel_size(), self.m_size, self.m_size],
                name='minimap_features'
            ),
            "screen": tf.placeholder(
                tf.float32, [None, util.screen_channel_size(), self.s_size, self.s_size],
                name='screen_features'
            ),
            "info": tf.placeholder(
                tf.float32, [None, util.structured_channel_size()],
                name='info_features'
            )
        }

        # Target placeholders
        self.targets = tf.placeholder(
                tf.float32, [None],
                name='value'
            )

        with tf.variable_scope("shared", reuse=reuse):
            _, self.fc = self.network.build(self.features)

        with tf.variable_scope("value_net"):
            self.prediction = layers.fully_connected(
                self.fc,
                num_outputs=1,
                activation_fn=None,
                scope='value'
            )

            self.losses = tf.squared_difference(self.prediction, self.targets)
            self.loss = tf.reduce_sum(self.losses, name="value_loss")

            if trainable:
                self.learning_rate = 1e-6  # tf.placeholder(tf.float32, None, name="learning_rate")
                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[g, v] for g, v in self.grads_and_vars if g is not None]
                self.train_op = self.optimizer.apply_gradients(
                    self.grads_and_vars,
                    global_step=tf.train.get_global_step(),
                    name="value_grads"
                )
