import numpy as np
import json

from pysc2.lib import features, actions

FEATURE_CONFIG = json.load(open('feature_config.json'))
INCLUDED_FEATURES = FEATURE_CONFIG["features"]
FORCED_SCALARS = FEATURE_CONFIG["forced_scalars"]

SCALAR = features.FeatureType.SCALAR
CATEGORICAL = features.FeatureType.CATEGORICAL

NON_SPATIAL_FEATURES = dict(
    player=(11,),
    game_loop=(1,),
    score_cumulative=(13,),
    available_actions=(len(actions.FUNCTIONS),),
    single_select=(1, 7),
    multi_select=(400, 7),  # (0, 7)
    cargo=(8, 7),  # (0, 7),
    cargo_slots_available=(1,),
    build_queue=(5, 7),  # (0, 7),
    control_groups=(10, 2),
)

screen_features = [x.name for x in features.SCREEN_FEATURES]
minimap_features = [x.name for x in features.MINIMAP_FEATURES]
non_spatial_features = [x for x in NON_SPATIAL_FEATURES.keys()]


def spatial_feature_channel_size(feature_layers_info, included_features, forced_scalars):
    """
    Args:
        feature_layers_info: pySC2 definitions for Minimap or Screen features
        forced_scalars: A list containing categorical feature layers to be treated as scalars
        included_features: A list containing features to include
    Returns:
        The channel size of the feature layers in feature_layers_info
    """
    c = 0
    for feature in feature_layers_info:
        if feature.name in included_features:
            if feature.type == SCALAR or feature.name in forced_scalars:
                c += 1
            else:
                c += feature.scale
    return c


def non_spatial_feature_channel_size(included_features):
    """
    Args:
        included_features: A list containing features to include
    Returns:
        The channel size of the non spatial features
    """
    c = 0
    for feature_name, shape in NON_SPATIAL_FEATURES.items():
        if feature_name in included_features:
            if len(shape) == 2:
                c += shape[0] * shape[1]
            else:
                c += shape[0]
    return c


SCREEN_FEATURE_SIZE = spatial_feature_channel_size(
    features.SCREEN_FEATURES,
    INCLUDED_FEATURES["screen"],
    FORCED_SCALARS["screen"]
)

MINIMAP_FEATURE_SIZE = spatial_feature_channel_size(
    features.MINIMAP_FEATURES,
    INCLUDED_FEATURES["minimap"],
    FORCED_SCALARS["minimap"]
)

NON_SPATIAL_FEATURE_SIZE = non_spatial_feature_channel_size(
    INCLUDED_FEATURES["non_spatial"],
)


# According to O. Vinyals et al. (2017) 4.3: Input pre-processing:
#   Scalars:     log transformation
#   Categorical: one-hot encoding in channel dimension (followed by 1x1 convolution)
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
#   Max total channels: 33
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
#   Max total channels: 1907
def preprocess_spatial_features(feature_layers, feature_layers_info, forced_scalars, included_features):
    """
    Args:
        feature_layers: Minimap or Screen attributes of SC2 observation dictionary
        feature_layers_info: pySC2 definitions for Minimap or Screen features
        forced_scalars: A list containing categorical feature layers to be treated as scalars
        included_features: A list containing features to include
    Returns:
        An array of shape
            (scalar features + categorical feature scales, screen_size or minimap_size, screen_size or minimap_size)
    """
    layers = []
    assert feature_layers.shape[0] == len(feature_layers_info)
    for i, feature in enumerate(feature_layers_info):
        if feature.name in included_features:
            if feature.type == SCALAR:
                layers.append(np.ma.log(feature_layers[i:i+1]))
            elif feature.name in forced_scalars:
                layers.append(feature_layers[i:i+1]/feature.scale)
            elif feature.type == CATEGORICAL:
                layer = np.zeros([feature.scale, feature_layers.shape[1], feature_layers.shape[2]], dtype=np.float32)
                for j in range(feature.scale):
                    y, x = (feature_layers[i] == j).nonzero()
                    layer[j, y, x] = 1
                layers.append(layer)
    return np.concatenate(layers, axis=0)


def preprocess_non_spatial_features(obs):
    info = np.zeros([NON_SPATIAL_FEATURE_SIZE], dtype=np.float32)
    info_offset = 0

    for feature, shape in NON_SPATIAL_FEATURES.items():
        if feature in INCLUDED_FEATURES["non_spatial"]:
            categorical_indices = []
            if feature == "player":
                categorical_indices = [0]

            if feature == "available_actions":
                info[info_offset + obs.observation[feature]] = 1
                info_offset += len(actions.FUNCTIONS)
            else:
                info = append_info_obs(
                    info,
                    info_offset,
                    obs.observation[feature],
                    categorical_indices
                )
                if len(shape) == 2:
                    info_offset += shape[0] * shape[1]
                else:
                    info_offset += shape[0]

    return info


def append_info_obs(info, info_offset, observation, categorical_indices):
    for i, feature in enumerate(observation):
        if not np.isscalar(feature):
            for j, attribute in enumerate(feature):
                offset = info_offset + j + i * len(feature)
                if j not in categorical_indices and attribute > 0:
                    info[offset] = np.log(attribute)
                else:
                    info[offset] = attribute
        # Take the log of numerical data to keep numbers low
        elif i not in categorical_indices and feature > 0:
            info[info_offset + i] = np.log(feature)
        else:
            info[info_offset + i] = feature

    return info