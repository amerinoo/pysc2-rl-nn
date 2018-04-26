import tensorflow as tf
import tensorflow.contrib.layers as layers

from pysc2.lib import actions
NUM_ACTIONS = len(actions.FUNCTIONS)
# TODO: Add num actions as class parameter


class AtariNet(object):
    def __init__(self, m_size, s_size):
        # Make sure that screen resolution is equal to minimap resolution
        assert (m_size == s_size)
        self.m_size = m_size
        self.s_size = s_size

    def build(self, features):
        # Minimap convolutions
        m_conv = self.cnn_block(features["minimap"], scope="m")
        s_conv = self.cnn_block(features["screen"], scope="s")
        non_spatial = self.non_spatial_block(features["info"], scope="non_spatial")

        # Create the state representation by concatenating on the channel axis
        feat_fc = layers.fully_connected(
            tf.concat([
                layers.flatten(m_conv),
                layers.flatten(s_conv),
                non_spatial
            ], axis=1),
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="feat_fc",
        )

        spatial_x = self.spatial_block(
            feat_fc=feat_fc,
            size=self.s_size,
            reshape_shape=[-1, 1, self.s_size],
            tile_shape=[1, self.s_size, 1],
            scope='x'
        )
        spatial_y = self.spatial_block(
            feat_fc=feat_fc,
            size=self.s_size,
            reshape_shape=[-1, self.s_size, 1],
            tile_shape=[1, 1, self.s_size],
            scope='y'
        )

        spatial_action = layers.flatten(spatial_x * spatial_y)

        non_spatial_action = layers.fully_connected(
            feat_fc,
            num_outputs=NUM_ACTIONS,
            activation_fn=tf.nn.softmax,
            scope='non_spatial_action'
        )

        value = tf.reshape(
            layers.fully_connected(
                feat_fc,
                num_outputs=1,
                activation_fn=None,
                scope='value'
            ), [-1]
        )

        return spatial_action, non_spatial_action, value

    def cnn_block(self, feature, scope):
        preprocess = layers.conv2d(
            tf.transpose(feature, [0, 2, 3, 1]),
            num_outputs=1,
            kernel_size=1,
            stride=1,
            padding="SAME",
            scope="{}_preprocess".format(scope)
        )

        conv1 = layers.conv2d(
            preprocess,
            num_outputs=16,
            kernel_size=8,
            stride=4,
            padding="SAME",
            scope="{}_conv2".format(scope)
        )
        
        return layers.conv2d(
            conv1,
            num_outputs=32,
            kernel_size=4,
            stride=2,
            padding="SAME",
            scope="{}_conv2".format(scope)
        )

    def spatial_block(self, feat_fc, size, reshape_shape, tile_shape, scope):
        return tf.tile(
            tf.reshape(
                layers.fully_connected(
                    feat_fc,
                    num_outputs=size,
                    activation_fn=tf.nn.softmax,
                    scope='spatial_{}'.format(scope)
                ),
                reshape_shape
            ),
            tile_shape
        )

    def non_spatial_block(self, feature, scope):
        return layers.fully_connected(
            layers.flatten(feature),
            num_outputs=256,
            activation_fn=tf.tanh,
            scope="{}_tanh_fc".format(scope)
        )
