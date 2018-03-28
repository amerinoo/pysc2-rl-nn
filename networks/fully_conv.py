import tensorflow as tf
import tensorflow.contrib.layers as layers

import util


class FullyConv(object):
    def __init__(self, m_size, s_size):
        # Make sure that screen resolution is equal to minimap resolution
        assert (m_size == s_size)
        self.m_size = m_size
        self.s_size = s_size

    def build(self, features, summarize=False):
        # Extract features while preserving the dimensions

        # Minimap convolutions
        m_preprocess = layers.conv2d(
            tf.transpose(features["minimap"], [0, 2, 3, 1]),
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
        s_preprocess = layers.conv2d(
            tf.transpose(features["screen"], [0, 2, 3, 1]),
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
        state_representation = tf.concat([
            m_conv2,
            s_conv2,
            tf.reshape(features["info"], [-1, self.s_size, self.s_size, 1])
        ], axis=3, name="state_representation")

        fc = layers.fully_connected(
            layers.flatten(state_representation),
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope='fully_conv_features',
        )

        if summarize:
            layers.summarize_activation(m_preprocess)
            layers.summarize_activation(m_conv1)
            layers.summarize_activation(m_conv2)
            layers.summarize_activation(s_conv1)
            layers.summarize_activation(s_conv2)
            layers.summarize_activation(fc)

        return state_representation, fc

