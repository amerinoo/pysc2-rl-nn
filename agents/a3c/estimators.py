import tensorflow as tf
import tensorflow.contrib.layers as layers
import pysc2.lib.actions as actions


class PolicyEstimator(object):
    def __init__(self, state, fc, summarize=False):
        self.state = state
        self.fc = fc

        self.summaries = []

        # Target placeholders
        self.advantages = tf.placeholder(
            tf.float32, [None],
            name='value'
        )

        # Valid action masks
        self.valid = {
            "spatial": tf.placeholder(
                tf.float32, [None],
                name='valid_spatial_actions'
            ),
            "non_spatial": tf.placeholder(
                tf.float32, [None, len(actions.FUNCTIONS)],
                name='valid_non_spatial_actions'
            )
        }

        # Action placeholders
        self.actions = {
            "spatial": tf.placeholder(
                tf.float32, [None, 64**2],    # TODO: Temporary constant
                # tf.int32, [None],
                name='spatial_action_selected'
            ),
            "non_spatial": tf.placeholder(
                tf.float32, [None, len(actions.FUNCTIONS)],
                # tf.int32, [None],
                name='non_spatial_action_selected'
            )
        }

        # Batch size = number of steps fed to network
        batch_size = tf.shape(self.advantages)[0]

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
                                scope='spatial_policy'
                            )
                        )
                    ),
                "non_spatial":
                    layers.fully_connected(
                        self.fc,
                        num_outputs=len(actions.FUNCTIONS),
                        activation_fn=tf.nn.softmax,
                        scope='non_spatial_policy'
                    )
            }

            # H(π) = -Σ(π(s) * log(π(s))) : over batched states
            # Clipping is done to prevent /0 and log(0)
            self.clipped_prediction = {
                "spatial": tf.clip_by_value(self.prediction["spatial"], 1e-10, 1.),
                "non_spatial": tf.clip_by_value(self.prediction["non_spatial"], 1e-10, 1.)
            }

            self.spatial_entropy = -tf.reduce_sum(
                self.prediction["spatial"] * tf.log(self.clipped_prediction["spatial"]), 1,
                name="spatial_entropy"
            )
            self.non_spatial_entropy = -tf.reduce_sum(
                self.prediction["non_spatial"] * tf.log(self.clipped_prediction["non_spatial"]), 1,
                name="non_spatial_entropy"
            )

            self.entropy_mean = tf.reduce_mean(
                [
                    tf.reduce_mean(self.spatial_entropy),
                    tf.reduce_mean(self.non_spatial_entropy)
                ],
                name="entropy_mean"
            )

            # Mask taken spatial_action probabilities
            self.spatial_probs = tf.reduce_sum(
                self.prediction["spatial"] * self.actions["spatial"],
                axis=1
            )
            self.spatial_probs_log = tf.log(tf.clip_by_value(self.spatial_probs, 1e-10, 1.))

            # Mask taken non_spatial_action probabilities
            self.non_spatial_probs = tf.reduce_sum(
                self.prediction["non_spatial"] * self.actions["non_spatial"],
                axis=1
            )

            # Mask invalid action probabilities
            self.valid_non_spatial_probs = tf.reduce_sum(
                self.prediction["non_spatial"] * self.valid["non_spatial"],
                axis=1
            )
            self.non_spatial_probs = self.non_spatial_probs * self.valid_non_spatial_probs
            self.non_spatial_probs_log = tf.log(tf.clip_by_value(self.non_spatial_probs, 1e-10, 1.))

            self.action_probs_log = self.valid["spatial"] * self.spatial_probs_log * self.non_spatial_probs_log

            # Policy Loss: L = -(log(π(s)) * A(s)) - β*H(π) : over batched states
            self.loss = tf.reduce_mean(self.action_probs_log * self.advantages) + 0.01 * self.entropy_mean

        if summarize:
            self.summaries.append(tf.summary.histogram('spatial_action_policy', self.clipped_prediction["spatial"]))
            self.summaries.append(tf.summary.histogram('non_spatial_action_policy', self.clipped_prediction["non_spatial"]))
            self.summaries.append(tf.summary.scalar('spatial_entropy', tf.reduce_mean(self.spatial_entropy)))
            self.summaries.append(tf.summary.scalar('non_spatial_entropy', tf.reduce_mean(self.non_spatial_entropy)))
            self.summaries.append(tf.summary.scalar('entropy', self.entropy_mean))
            # self.summaries.append(tf.summary.histogram('spatial_loss', self.spatial_losses))
            # self.summaries.append(tf.summary.histogram('non_spatial_loss', self.non_spatial_losses))
            self.summaries.append(tf.summary.scalar('policy_loss', self.loss))
            self.summaries = tf.summary.merge(self.summaries)


class ValueEstimator(object):
    def __init__(self, fc, summarize=False):
        self.fc = fc

        self.summaries = []

        # Target placeholders
        self.targets = tf.placeholder(
                tf.float32, [None],
                name='value'
            )

        with tf.variable_scope("value_net"):
            self.prediction = layers.fully_connected(
                self.fc,
                num_outputs=1,
                activation_fn=None,
                scope='value'
            )

            self.losses = tf.squared_difference(self.prediction, self.targets)
            self.loss = tf.reduce_mean(self.losses, name="value_loss")

            if summarize:
                self.summaries.append(tf.summary.scalar("max_value", tf.reduce_max(self.prediction)))
                self.summaries.append(tf.summary.scalar("min_value", tf.reduce_min(self.prediction)))
                self.summaries.append(tf.summary.scalar("mean_value", tf.reduce_mean(self.prediction)))
                self.summaries.append(tf.summary.scalar("reward_max", tf.reduce_max(self.targets)))
                self.summaries.append(tf.summary.scalar("reward_min", tf.reduce_min(self.targets)))
                self.summaries.append(tf.summary.scalar("reward_mean", tf.reduce_mean(self.targets)))
                self.summaries.append(tf.summary.histogram("reward_targets", self.targets))
                self.summaries.append(tf.summary.histogram("values", self.prediction))

        if summarize:
            self.summaries.append(tf.summary.scalar('value_loss', self.loss))
            self.summaries = tf.summary.merge(self.summaries)


class Optimizer(object):
    def __init__(self, name, learning_rate, loss):
        self.name = name
        self.learning_rate = learning_rate
        self.loss = loss

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate, decay=0.99, epsilon=1e-6,
            name="{}_optimizer".format(self.name)
        )

        self.grads_and_vars = self.optimizer.compute_gradients(loss)
        self.grads_and_vars = [[g, v] for g, v in self.grads_and_vars if g is not None]
        self.train_op = self.optimizer.apply_gradients(
            self.grads_and_vars,
            global_step=tf.train.get_global_step(),
            name="{}_grads".format(self.name)
        )
