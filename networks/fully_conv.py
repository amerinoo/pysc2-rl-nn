import tensorflow as tf
import tensorflow.contrib.layers as layers
import util
import logging

logging.basicConfig(level=logging.INFO)


class FullyConv(object):
    def __init__(self, m_size, s_size, num_actions):
        # Make sure that screen resolution is equal to minimap resolution
        assert (m_size == s_size)
        self.m_size = m_size
        self.s_size = s_size
        self.num_actions = num_actions

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

        self.policy_net = {}
        self.value_net = None

    def build(self):
        logging.info("Building network...")
        # Extract features while preserving the dimensions

        # Minimap convolutions
        m_preprocess = layers.conv2d(
            tf.transpose(self.features["minimap"], [0, 2, 3, 1]),
            num_outputs=1,
            kernel_size=1,
            stride=1,
            padding="SAME",
            scope="m_preprocess"
        )

        m_conv1 = layers.conv2d(
            m_preprocess,
            num_outputs=16,
            kernel_size=5,
            stride=1,
            padding="SAME",
            scope="m_conv1"
        )

        m_conv2 = layers.conv2d(
            m_conv1,
            num_outputs=32,
            kernel_size=3,
            stride=1,
            padding="SAME",
            scope="m_conv2"
        )

        # Screen convolutions
        s_preprocess = layers.conv2d(tf.transpose(
            self.features["screen"], [0, 2, 3, 1]),
            num_outputs=1,
            kernel_size=1,
            stride=1,
            padding="SAME",
            scope="s_preprocess"
        )

        s_conv1 = layers.conv2d(
            s_preprocess,
            num_outputs=16,
            kernel_size=5,
            stride=1,
            padding="SAME",
            scope="s_conv1"
        )

        s_conv2 = layers.conv2d(
            s_conv1,
            num_outputs=32,
            kernel_size=3,
            stride=1,
            padding="SAME",
            scope="s_conv2"
        )

        # Create the state representation by concatenating on the channel axis
        state = tf.concat([
            m_conv2,
            s_conv2,
            tf.reshape(self.features["info"], [-1, self.s_size, self.s_size, 1])
        ], axis=3)

        # Perform another convolution, but preserve the dimensions by using params(1, 1, 1)
        spatial_action_policy = layers.conv2d(
            state,
            num_outputs=1,
            kernel_size=1,
            stride=1,
            activation_fn=None,
            weights_initializer=util.normalized_columns_initializer(1.0),
            scope='spatial_features'
        )

        # spatial_action policy
        self.policy_net["spatial"] = tf.nn.softmax(layers.flatten(spatial_action_policy))

        feat_fc = layers.fully_connected(
            layers.flatten(state),
            num_outputs=256,
            activation_fn=tf.nn.relu,
            weights_initializer=util.normalized_columns_initializer(1.0),
            scope='feat_fc'
        )

        # non_spatial_action policy
        self.policy_net["non_spatial"] = layers.fully_connected(
            feat_fc,
            num_outputs=self.num_actions,
            activation_fn=tf.nn.softmax,
            weights_initializer=util.normalized_columns_initializer(
                1.0),
            scope='non_spatial_action'
        )

        # Value
        self.value_net = layers.fully_connected(
            feat_fc,
            num_outputs=1,
            activation_fn=None,
            weights_initializer=util.normalized_columns_initializer(1.0),
            scope='value'
        )

        logging.info("Network built!")
        return self.policy_net, self.value_net
