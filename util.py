import numpy as np
import tensorflow as tf

import importlib

from pysc2.lib import features, actions

import math

_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index


def init_network(network, m_size, s_size, num_actions):
    network_module, network_name = network.rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module), network_name)

    return network_cls(m_size, s_size, num_actions)


def minimap_obs(obs):
    m = np.array(obs.observation["minimap"], dtype=np.float32)
    return np.expand_dims(preprocess_minimap(m), axis=0)


def screen_obs(obs):
    s = np.array(obs.observation["screen"], dtype=np.float32)
    return np.expand_dims(preprocess_screen(s), axis=0)


def info_obs(obs):
    info = np.zeros([1, structured_channel_size()], dtype=np.float32)

    # Mask available actions
    info[0, obs.observation["available_actions"]] = 1

    # General player information:
    #     player_id, minerals, vespene, food_used, food_cap, food_army,
    #     food_workers, idle_worker_count, army_count, warp_gate_count, larva_count,
    player_obs_len = len(obs.observation["player"])
    for offset in range(player_obs_len):
        feature = obs.observation["player"][offset]

        # Take the log of numerical data to keep numbers low
        if offset > 0 and feature > 0:
            info[0, len(actions.FUNCTIONS) + offset] = math.log(feature)
        else:
            info[0, len(actions.FUNCTIONS) + offset] = feature

    # Single select information:
    #     unit_type, player_relative, health, shields,
    #     energy, transport_slots_taken, build_progress
    single_obs_len = len(obs.observation["single_select"][0])
    for offset in range(single_obs_len):
        feature = obs.observation["single_select"][0][offset]
        # Take the log of numerical data to keep numbers low
        if ((2 <= offset <= 4) or offset == 6) and feature > 0:
            info[0, len(actions.FUNCTIONS) + player_obs_len + offset] = math.log(feature)
        else:
            info[0, len(actions.FUNCTIONS) + player_obs_len + offset] = feature

    # TODO:
    #   Add control groups
    #   Add multi-select
    #   Add cargo
    #   Add build_queue

    return info


def preprocess_minimap(minimap):
    layers = []
    assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
    for i in range(len(features.MINIMAP_FEATURES)):
        if i == _MINIMAP_PLAYER_ID:
            layers.append(minimap[i:i + 1] / features.MINIMAP_FEATURES[i].scale)
        elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
            layers.append(minimap[i:i + 1] / features.MINIMAP_FEATURES[i].scale)
        else:
            layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)
            for j in range(features.MINIMAP_FEATURES[i].scale):
                indy, indx = (minimap[i] == j).nonzero()
                layer[j, indy, indx] = 1
            layers.append(layer)
    return np.concatenate(layers, axis=0)


def preprocess_screen(screen):
    layers = []
    assert screen.shape[0] == len(features.SCREEN_FEATURES)
    for i in range(len(features.SCREEN_FEATURES)):
        if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
            layers.append(screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)
        elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
            layers.append(screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)
        else:
            layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
            for j in range(features.SCREEN_FEATURES[i].scale):
                indy, indx = (screen[i] == j).nonzero()
                layer[j, indy, indx] = 1
            layers.append(layer)
    return np.concatenate(layers, axis=0)


def minimap_channel_size():
    c = 0
    for i in range(len(features.MINIMAP_FEATURES)):
        if i == _MINIMAP_PLAYER_ID:
            c += 1
        elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
            c += 1
        else:
            c += features.MINIMAP_FEATURES[i].scale
    return c


def screen_channel_size():
    c = 0
    for i in range(len(features.SCREEN_FEATURES)):
        if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
            c += 1
        elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
            c += 1
        else:
            c += features.SCREEN_FEATURES[i].scale
    return c


def structured_channel_size():
    # Upper bound for structured data
    return 4096


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def get_action_arguments(act_id, target):
    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
        if arg.name in ('screen', 'minimap', 'screen2'):
            act_args.append([target[1], target[0]])
            pass
        else:
            act_args.append([0])

    return act_args
