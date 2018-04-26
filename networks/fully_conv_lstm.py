import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn
# import numpy as np


class FullyConvLSTM(object):
    def __init__(self, m_size, s_size):
        # Make sure that screen resolution is equal to minimap resolution
        assert (m_size == s_size)
        self.m_size = m_size
        self.s_size = s_size

    def build(self, features):
        # Extract features while preserving the dimensions

        # Minimap convolutions
        m_conv = cnn_block(features["minimap"], scope="m")
        s_conv = cnn_block(features["screen"], scope="s")

        # Create the state representation by concatenating on the channel axis
        state_representation = tf.concat([
            m_conv,
            s_conv,
            tf.transpose(features["info"], [0, 2, 3, 1])
        ], axis=3, name="state_representation")

        state_shape = state_representation.shape.as_list()

        conv_LSTM_cell = rnn.ConvLSTMCell(
            conv_ndims=1,
            input_shape=[state_shape[1], state_shape[2]],
            output_channels=state_shape[3],
            kernel_shape=[1]
        )

        conv_LSTM_out, state = tf.nn.dynamic_rnn(
            conv_LSTM_cell,
            state_representation,
            dtype=state_representation.dtype
        )

        fc = layers.fully_connected(
            layers.flatten(conv_LSTM_out),
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope='fully_conv_lstm_features',
        )

        spatial_action = tf.nn.softmax(
            layers.flatten(
                layers.conv2d(
                    state_representation,
                    num_outputs=1,
                    kernel_size=1,
                    stride=1,
                    activation_fn=None,
                    scope='spatial_policy'
                )
            )
        )

        non_spatial_action = layers.fully_connected(
            fc,
            num_outputs=len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope='non_spatial_policy'
        )

        value = layers.fully_connected(
            fc,
            num_outputs=1,
            activation_fn=None,
            scope='value'
        )

        return spatial_action, non_spatial_action, value


def cnn_block(feature, scope):
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
        kernel_size=5,
        stride=1,
        padding="SAME",
        scope="{}_conv1".format(scope)
    )

    conv2 = layers.conv2d(
        conv1,
        num_outputs=32,
        kernel_size=3,
        stride=1,
        padding="SAME",
        scope="{}_conv2".format(scope)
    )

    return conv2
