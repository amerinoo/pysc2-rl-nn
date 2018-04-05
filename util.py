import numpy as np
import tensorflow as tf

import importlib

from pysc2.lib import features, actions


def spatial_features_channel_size(feature_layers_info, forced_scalars=None, excluded_features=None):
    """
    Args:
        feature_layers_info: pySC2 definitions for Minimap or Screen features
        forced_scalars: A list containing categorical feature layers to be treated as scalars
        excluded_features: A list containing features to exclude
    Returns:
        The channel size of the feature layers in feature_layers_info
    """
    if forced_scalars is None:
        forced_scalars = []
    if excluded_features is None:
        excluded_features = []
    c = 0
    for i in range(len(feature_layers_info)):
        if i in excluded_features:
            continue
        if feature_layers_info[i].type == features.FeatureType.SCALAR or i in forced_scalars:
            c += 1
        else:
            c += feature_layers_info[i].scale
    return c


def structured_channel_size():
    # "player": (11,),
    c = 11
    # "game_loop": (1,),
    # c += 1
    # "score_cumulative": (13,),
    # c += 13
    # "available_actions": (0,),
    c += len(actions.FUNCTIONS)  # 524
    # "single_select": (0, 7), Actually only (n, 7) for n in (0, 1)
    c += 7
    # "multi_select": (0, 7),
    c += 7 * 400  # Technical unit selection limit is 500, 400 is the practical limit (excluding Overlords)
    # "cargo": (0, 7),
    # c += 7 * 8  # Max cargo
    # "cargo_slots_available": (1,),
    # c += 1
    # "build_queue": (0, 7),
    # c += 7 * 5  # Max build queue
    # "control_groups": (10, 2),
    # c += 10 * 2
    c = (c + 1024 - (c % 1024))
    # return c
    return 4096


_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

_MINIMAP_FEATURE_SIZE = spatial_features_channel_size(
    features.MINIMAP_FEATURES,
    forced_scalars=[_MINIMAP_PLAYER_ID],
    excluded_features=[2])
_SCREEN_FEATURE_SIZE = spatial_features_channel_size(
    features.SCREEN_FEATURES,
    forced_scalars=[_SCREEN_PLAYER_ID, _SCREEN_UNIT_TYPE],
    excluded_features=[2, 3, 10, 11, 12, 13])
_STRUCTURED_FEATURE_SIZE = structured_channel_size()



def init_network(network, m_size, s_size):
    network_module, network_name = network.rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module), network_name)

    return network_cls(m_size, s_size)


def init_feature_placeholders(m_size, s_size):
    return {
        "minimap": tf.placeholder(
            tf.float32, [None, _MINIMAP_FEATURE_SIZE, m_size, m_size],
            name='minimap_features'
        ),
        "screen": tf.placeholder(
            tf.float32, [None, _SCREEN_FEATURE_SIZE, s_size, s_size],
            name='screen_features'
        ),
        "info": tf.placeholder(
            tf.float32, [None, _STRUCTURED_FEATURE_SIZE],
            name='info_features'
        )
    }


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
        preprocess_spatial_features(
            m,
            features.MINIMAP_FEATURES,
            forced_scalars=[_MINIMAP_PLAYER_ID],
            excluded_features=[2]
        ),
        axis=0
    )


def screen_obs(obs):
    """
    Args:
        obs: Observation from the SC2 environment
    Returns:
        Give no forced scalars or excluded features,
        screen feature layers of shape (1, 1907, screen_size, screen_size)
    """
    s = np.array(obs.observation["screen"], dtype=np.float32)   # shape = (17, size_s, size_s)
    return np.expand_dims(
        preprocess_spatial_features(
            s,
            features.SCREEN_FEATURES,
            forced_scalars=[_SCREEN_PLAYER_ID, _SCREEN_UNIT_TYPE],
            excluded_features=[2, 3, 10, 11, 12, 13]
        ),
        axis=0
    )


def info_obs(obs):
    info = np.zeros([1, structured_channel_size()], dtype=np.float32)
    info_offset = 0

    info, info_offset = append_info_obs(
        info,
        info_offset,
        obs.observation["player"],
        categorical_indices=[0]
    )

    # info, info_offset = append_info_obs(
    #     info,
    #     info_offset,
    #     obs.observation["game_loop"]
    # )
    #
    # info, info_offset = append_info_obs(
    #     info,
    #     info_offset,
    #     obs.observation["score_cumulative"]
    # )

    info[0, info_offset + obs.observation["available_actions"]] = 1
    info_offset += len(actions.FUNCTIONS)

    info, _ = append_info_obs(
        info,
        info_offset,
        obs.observation["single_select"]
        # categorical_indices=[0, 1]
    )

    info_offset += 7

    # info, _ = append_info_obs(
    #     info,
    #     info_offset,
    #     obs.observation["multi_select"]
    #     # categorical_indices=[0, 1]
    # )

    # info_offset += 7*400
    #
    # info, _ = append_info_obs(
    #     info,
    #     info_offset,
    #     obs.observation["cargo"]
    #     # categorical_indices=[0, 1]
    # )
    #
    # info_offset += 7*8
    #
    # info, info_offset = append_info_obs(
    #     info,
    #     info_offset,
    #     obs.observation["cargo_slots_available"]
    #     # categorical_indices=[0, 1]
    # )
    #
    # info, _ = append_info_obs(
    #     info,
    #     info_offset,
    #     obs.observation["build_queue"]
    #     # categorical_indices=[0, 1]
    # )
    #
    # info_offset += 7*5
    #
    # info, info_offset = append_info_obs(
    #     info,
    #     info_offset,
    #     obs.observation["control_groups"]
    #     # categorical_indices=[0, 1]
    # )

    return info


def append_info_obs(info, info_offset, observation, categorical_indices=None):
    if categorical_indices is None:
        categorical_indices = []
    for i, feature in enumerate(observation):
        if not np.isscalar(feature):
            for j, attribute in enumerate(feature):
                offset = info_offset + j + i * len(feature)
                if j not in categorical_indices and attribute > 0:
                    info[0, offset] = np.log(attribute)
                else:
                    info[0, offset] = attribute
        # Take the log of numerical data to keep numbers low
        elif i not in categorical_indices and feature > 0:
            info[0, info_offset + i] = np.log(feature)
        else:
            info[0, info_offset + i] = feature
    info_offset += len(observation)

    return info, info_offset


# According to O. Vinyals et al. (2017) 4.3: Input pre-processing:
#   Scalars:     log transformation
#   Categorical: one-hot encoding in channel dimension
#
# Minimap Features:
#     feature:          (scale, type)
#     height_map:       (256,   SCALAR)
#     visibility_map:   (4,     CATEGORICAL)
#     creep:            (2,     CATEGORICAL)
#     camera:           (2,     CATEGORICAL)
#     player_id:        (17,    CATEGORICAL)
#     player_relative:  (5,     CATEGORICAL)
#     selected:         (2,     CATEGORICAL)
#
#   Total channels: 33
#
# Screen Features:
#     feature:              (scale, type)
#     height_map:           (256,   SCALAR)
#     visibility_map:       (4,     CATEGORICAL)
#     creep:                (2,     CATEGORICAL)
#     power:                (2,     CATEGORICAL)
#     player_id:            (17,    CATEGORICAL)
#     player_relative:      (5,     CATEGORICAL)
#     unit_type:            (1850,  CATEGORICAL)
#     selected:             (2,     CATEGORICAL)
#     unit_hit_points:      (1600,  SCALAR)
#     unit_hit_points_ratio:(256,   SCALAR)
#     unit_energy:          (1000,  SCALAR)
#     unit_energy_ratio:    (256,   SCALAR)
#     unit_shields:         (1000,  SCALAR)
#     unit_shields_ratio:   (256,   SCALAR)
#     unit_density:         (16,    SCALAR)
#     unit_density_aa:      (256,   SCALAR)
#     effects:              (16,    CATEGORICAL)
#
#   Total channels: 1907
def preprocess_spatial_features(feature_layers, feature_layers_info, forced_scalars=None, excluded_features=None):
    """
    Args:
        feature_layers: Minimap or Screen attributes of SC2 observation dictionary
        feature_layers_info: pySC2 definitions for Minimap or Screen features
        forced_scalars: A list containing categorical feature layers to be treated as scalars
        excluded_features: A list containing features to exclude
    Returns:
        An array of shape
            (scalar features + categorical feature scales, screen_size or minimap_size, screen_size or minimap_size)
    """
    if forced_scalars is None:
        forced_scalars = []
    if excluded_features is None:
        excluded_features = []
    layers = []
    assert feature_layers.shape[0] == len(feature_layers_info)
    for i, feature_layer in enumerate(feature_layers_info):
        if i in excluded_features:
            continue
        if feature_layer.type == features.FeatureType.SCALAR or i in forced_scalars:
            layers.append(np.ma.log(feature_layers[i:i+1]))
        elif i in forced_scalars:
            layers.append(feature_layers[i:i+1]/feature_layer.scale)
        elif feature_layer.type == features.FeatureType.CATEGORICAL:
            layer = np.zeros([feature_layer.scale, feature_layers.shape[1], feature_layers.shape[2]], dtype=np.float32)
            for j in range(feature_layer.scale):
                y, x = (feature_layers[i] == j).nonzero()
                layer[j, y, x] = 1
            layers.append(layer)
    return np.concatenate(layers, axis=0)


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