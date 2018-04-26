import importlib

import numpy as np
import tensorflow as tf

from pysc2.lib import actions, features

from common import preprocessing as pp

SCREEN_FEATURE_SIZE = pp.SCREEN_FEATURE_SIZE
MINIMAP_FEATURE_SIZE = pp.MINIMAP_FEATURE_SIZE
NON_SPATIAL_FEATURE_SIZE = pp.NON_SPATIAL_FEATURE_SIZE

NON_SPATIAL_FEATURES = pp.NON_SPATIAL_FEATURES

FORCED_SCALARS = pp.FORCED_SCALARS
INCLUDED_FEATURES = pp.INCLUDED_FEATURES


def minimap_obs(obs):
    """
    Args:
        obs: Observation from the SC2 environment
    Returns:
        Given no forced scalars or excluded features,
        minimap feature layers of shape (1, 33, minimap_size, minimap_size)
    """
    m = np.array(obs.observation["minimap"], dtype=np.float32)  # shape = (7, size_m, size_m)
    return np.expand_dims(
        pp.preprocess_spatial_features(
            m,
            features.MINIMAP_FEATURES,
            FORCED_SCALARS["minimap"],
            INCLUDED_FEATURES["minimap"]
        ),
        axis=0
    )


def screen_obs(obs):
    """
    Args:
        obs: Observation from the SC2 environment
    Returns:
        Given no forced scalars or excluded features,
        screen feature layers of shape (1, 1907, screen_size, screen_size)
    """
    s = np.array(obs.observation["screen"], dtype=np.float32)  # shape = (17, size_s, size_s)
    return np.expand_dims(
        pp.preprocess_spatial_features(
            s,
            features.SCREEN_FEATURES,
            FORCED_SCALARS["screen"],
            INCLUDED_FEATURES["screen"]
        ),
        axis=0
    )


def non_spatial_obs(obs, size):
    non_spatial = pp.preprocess_non_spatial_features(obs)

    # repeat vector stats over (-1, 1, 64, 64)
    min_shape = non_spatial.shape[0]
    max_shape = size**2
    last_index = max_shape - max_shape % min_shape
    repeats = max_shape//min_shape

    out_non_spatial = np.zeros(max_shape)
    out_non_spatial[:last_index] = np.concatenate(np.repeat([non_spatial], repeats, axis=0))
    return np.reshape(out_non_spatial, [-1, 1, size, size])


def init_network(network, m_size, s_size):
    network_module, network_name = network.rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module), network_name)

    return network_cls(m_size, s_size)


def init_feature_placeholders(m_size, s_size):
    return {
        "minimap": tf.placeholder(
            tf.float32, [None, MINIMAP_FEATURE_SIZE, m_size, m_size],
            name='minimap_features'
        ),
        "screen": tf.placeholder(
            tf.float32, [None, SCREEN_FEATURE_SIZE, s_size, s_size],
            name='screen_features'
        ),
        "info": tf.placeholder(
            tf.float32, [None, 1, s_size, s_size],
            name='non_spatial_features'
        )
    }

# def init_non_spatial_placeholder(size):
#     dims = get_valid_dims(NON_SPATIAL_FEATURES, INCLUDED_FEATURES["non_spatial"])
#
#     tensors = [tf.placeholder(tf.float32, [None, 1, 1, np.prod(dim)]) for dim in dims]
#     tensors = tf.concat(tensors, axis=3)
#     tensors = tf.tile(tensors, [1, size, size, 1])
#     return tensors


def get_valid_dims(feats, included_features):
    return [dim for f, dim in feats.items() if f in included_features]


def get_action_arguments(act_id, target):
    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
        if arg.name in ('screen', 'minimap', 'screen2'):
            act_args.append([target[1], target[0]])
            pass
        else:
            act_args.append([0])

    return act_args


def make_copy_params_op(v1_list, v2_list):
    v1_list = list(sorted(v1_list, key=lambda v: v.name))
    v2_list = list(sorted(v2_list, key=lambda v: v.name))

    update_ops = []
    for v1, v2 in zip(v1_list, v2_list):
        try:
            op = v2.assign(v1)
            update_ops.append(op)
        except ValueError as e:
            print(e)

    return update_ops


def make_train_op(local_optimizer, global_optimizer):
    local_grads, _ = zip(*local_optimizer.grads_and_vars)
    # Clip gradients
    local_grads, _ = tf.clip_by_global_norm(local_grads, 1.0)
    _, global_vars = zip(*global_optimizer.grads_and_vars)

    local_global_grads_and_vars = list(zip(local_grads, global_vars))

    return global_optimizer.optimizer.apply_gradients(
        local_global_grads_and_vars,
        global_step=tf.train.get_global_step()
    )
