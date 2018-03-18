import tensorflow as tf
import tensorflow.contrib.layers as layers
import util
import pysc2.lib.actions as actions


class Estimator(object):
    def __init__(self, s_size):
        # Input placeholders
        self.input = tf.placeholder(shape=[None, 64, 64, 33], dtype=tf.float32, name="state")

        # Target placeholders
        self.targets = {
            "spatial": tf.placeholder(
                tf.float32, [None, s_size ** 2],
                name='spatial_action_selected'
            ),
            "non_spatial": tf.placeholder(
                tf.float32, [None, len(actions.FUNCTIONS)],
                name='non_spatial_action_selected'
            ),
            "value": tf.placeholder(
                tf.float32, [None],
                name='value_target'
            )
        }

        # Shared network
        self.fully_conv_features = None

    def build_shared_network(self, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            self.fully_conv_features = layers.fully_connected(
                layers.flatten(self.input),
                num_outputs=256,
                activation_fn=tf.nn.relu,
                weights_initializer=util.normalized_columns_initializer(1.0),
                scope='feat_fc'
            )

    def build_policy_network(self, scope):
        with tf.variable_scope(scope):
            self.targets["spatial"] = tf.nn.softmax(
                layers.flatten(
                    layers.conv2d(
                        self.input,
                        num_outputs=1,
                        kernel_size=1,
                        stride=1,
                        activation_fn=None,
                        weights_initializer=util.normalized_columns_initializer(1.0),
                        scope='spatial_target'
                    )
                )
            )

            # non_spatial_action policy
            self.targets["non_spatial"] = layers.fully_connected(
                self.fully_conv_features,
                num_outputs=len(actions.FUNCTIONS),
                activation_fn=tf.nn.softmax,
                weights_initializer=util.normalized_columns_initializer(1.0),
                scope='non_spatial_target'
            )

    def build_value_network(self, scope):
        with tf.variable_scope(scope):
            # Value
            self.targets["value"] = layers.fully_connected(
                self.fully_conv_features,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=util.normalized_columns_initializer(1.0),
                scope='value_target'
            )

