import tensorflow as tf
import tensorflow.contrib.layers as layers
# import tensorflow.contrib.rnn as rnn
# import numpy as np


class FullyConvLSTM(object):
    def __init__(self, m_size, s_size, num_action):
        self.features = {}

        # Make sure that screen resolution is equal to minimap resolution
        assert(m_size == s_size)
        self.m_size = m_size
        self.s_size = s_size

        self.num_action = num_action

    def init_inputs(self, features):
        self.features = features

    def build(self):
        # Extract features while preserving the dimensions

        # Minimap convolutions
        m_preprocess = layers.conv2d(tf.transpose(self.features["minimap"], [0, 2, 3, 1]),
                             num_outputs=1,
                             kernel_size=1,
                             stride=1,
                             padding="SAME",
                             scope="m_preprocess")

        m_conv1 = layers.conv2d(m_preprocess,
                               num_outputs=16,
                               kernel_size=5,
                               stride=1,
                               padding="SAME",
                               scope="m_conv1")

        m_conv2 = layers.conv2d(m_conv1,
                               num_outputs=32,
                               kernel_size=3,
                               stride=1,
                               padding="SAME",
                               scope="m_conv2")

        # Screen convolutions
        s_preprocess = layers.conv2d(tf.transpose(self.features["screen"], [0, 2, 3, 1]),
                             num_outputs=1,
                             kernel_size=1,
                             stride=1,
                             padding="SAME",
                             scope="s_preprocess")

        s_conv1 = layers.conv2d(s_preprocess,
                               num_outputs=16,
                               kernel_size=5,
                               stride=1,
                               padding="SAME",
                               scope="s_conv1")

        s_conv2 = layers.conv2d(s_conv1,
                               num_outputs=32,
                               kernel_size=3,
                               stride=1,
                               padding="SAME",
                               scope="s_conv2")

        # Create the state representation by concatenating on the channel axis
        state_representation = tf.concat([
            m_conv2,
            s_conv2,
            tf.reshape(self.features["info"], [-1, self.m_size, self.s_size, 1])
        ], axis=3)


        # LSTM Layer
        # lstm_cell = rnn.BasicLSTMCell(256, state_is_tuple=True)
        # c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        # h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        # self.state_init = [c_init, h_init]
        # c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        # h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        # self.state_in = (c_in, h_in)
        # rnn_in = tf.expand_dims(state_representation, [0])
        # step_size = tf.shape(m_pp)[:1]  # temp step size
        #
        # state_in = rnn.LSTMStateTuple(c_in, h_in)
        # lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        #     lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
        #     time_major=False)
        # lstm_c, lstm_h = lstm_state
        # self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        #
        # rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        # Perform another convolution, but preserve the dimensions by using params(1, 1, 1)
        spatial_action_policy = layers.conv2d(state_representation,
                                              num_outputs=1,
                                              kernel_size=1,
                                              stride=1,
                                              activation_fn=None,
                                              scope='spatial_feat')

        spatial_action = tf.nn.softmax(layers.flatten(spatial_action_policy))

        feat_fc = layers.fully_connected(layers.flatten(state_representation),
                                         num_outputs=self.m_size*self.s_size,
                                         activation_fn=tf.nn.relu,
                                         scope='feat_fc')

        non_spatial_action = layers.fully_connected(feat_fc,
                                                    num_outputs=self.num_action,
                                                    activation_fn=tf.nn.softmax,
                                                    scope='non_spatial_action')

        value = layers.fully_connected(feat_fc,
                                       num_outputs=1,
                                       activation_fn=None,
                                       scope='value')

        return {"spatial": spatial_action,
                "non_spatial": non_spatial_action,
                "value": value}
