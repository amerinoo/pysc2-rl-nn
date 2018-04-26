import tensorflow as tf
import tensorflow.contrib.layers as layers
import pysc2.lib.actions as actions


def configure_estimators(network, features, eta, beta, learning_rate, dual_msprop=False, summary_writer=None):
    # Obtain state representation and fullyConv from network
    spatial, non_spatial, value = network.build(features)

    # Build the rest of the network
    policy_net = PolicyEstimator(
        spatial, non_spatial, eta, summarize=summary_writer is not None)
    value_net = ValueEstimator(
        value, summarize=summary_writer is not None)
    # TODO: if is_training
    if dual_msprop:
        policy_optimizer = Optimizer("policy", learning_rate, policy_net.loss)
        value_optimizer = Optimizer("value", learning_rate, beta * value_net.loss)
        return policy_net, value_net, [policy_optimizer, value_optimizer]
    else:
        optimizer = Optimizer("single", learning_rate, policy_net.loss + beta * value_net.loss)
        return policy_net, value_net, [optimizer]


class PolicyEstimator(object):
    def __init__(self, spatial_action, non_spatial_action, eta, summarize=False):
        self.prediction = {
            "spatial": spatial_action,
            "non_spatial": non_spatial_action
        }

        self.eta = eta

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
                tf.float32, [None, 32**2],    # TODO: Temporary constant
                name='spatial_action_selected'
            ),
            "non_spatial": tf.placeholder(
                tf.float32, [None, len(actions.FUNCTIONS)],
                name='non_spatial_action_selected'
            )
        }

        with tf.variable_scope("policy_net"):
            # H(π) = Σ(π(s) * log(π(s))) : over batched states
            # Clipping is done to prevent /0 and log(0)
            self.clipped_prediction = {
                "spatial": tf.clip_by_value(self.prediction["spatial"], 1e-10, 1.),
                "non_spatial": tf.clip_by_value(self.prediction["non_spatial"], 1e-10, 1.)
            }

            self.spatial_entropy = tf.reduce_sum(
                self.prediction["spatial"] * tf.log(self.clipped_prediction["spatial"]), 1,
                name="spatial_entropy"
            )
            self.non_spatial_entropy = tf.reduce_sum(
                self.prediction["non_spatial"] * tf.log(self.clipped_prediction["non_spatial"]), 1,
                name="non_spatial_entropy"
            )

            self.entropy_mean = -tf.reduce_mean(
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
            ) * self.valid["spatial"]
            self.spatial_probs_log = tf.log(tf.clip_by_value(self.spatial_probs, 1e-10, 1.))

            # Mask taken non_spatial_action probabilities
            # Mask invalid action probabilities
            self.non_spatial_probs = tf.reduce_sum(
                self.prediction["non_spatial"] * self.actions["non_spatial"] * self.valid["non_spatial"],
                axis=1
            )
            self.non_spatial_probs_log = tf.log(tf.clip_by_value(self.non_spatial_probs, 1e-10, 1.))

            self.action_probs_log = self.spatial_probs_log * self.non_spatial_probs_log

            # Policy Loss: L = log(π(s) * A(s)) - β*H(π) : over batched states
            self.loss = (tf.reduce_mean(self.action_probs_log * self.advantages)) + self.eta * self.entropy_mean

        if summarize:
            self.summaries.append(tf.summary.histogram('spatial_action_policy', self.clipped_prediction["spatial"]))
            self.summaries.append(tf.summary.histogram('non_spatial_action_policy', self.clipped_prediction["non_spatial"]))
            self.summaries.append(tf.summary.scalar('spatial_entropy', tf.reduce_mean(self.spatial_entropy)))
            self.summaries.append(tf.summary.scalar('non_spatial_entropy', tf.reduce_mean(self.non_spatial_entropy)))
            self.summaries.append(tf.summary.scalar('entropy', self.entropy_mean))
            self.summaries.append(tf.summary.scalar('policy_loss', self.loss))
            self.summaries = tf.summary.merge(self.summaries)


class ValueEstimator(object):
    def __init__(self, value, summarize=False):
        self.prediction = value

        self.summaries = []

        # Target placeholders
        self.targets = tf.placeholder(
                tf.float32, [None],
                name='value'
            )

        with tf.variable_scope("value_net"):
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
            learning_rate,
            name="{}_optimizer".format(self.name)
        )

        self.grads_and_vars = self.optimizer.compute_gradients(loss)
        self.grads_and_vars = [[g, v] for g, v in self.grads_and_vars if g is not None]
        self.train_op = self.optimizer.apply_gradients(
            self.grads_and_vars,
            global_step=tf.train.get_global_step(),
            name="{}_grads".format(self.name)
        )
