import importlib
import os
import sys
import numpy as np

from pysc2 import maps
from pysc2.env import sc2_env
import tensorflow as tf

from absl import app
from absl import flags

import time

# Training Flags
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 2e-4, "Learning rate for training.")
flags.DEFINE_float("discount", 0.95, "Discount rate for future rewards.")
flags.DEFINE_float("eta", 1e-6, "Entropy regularization weight.")
flags.DEFINE_float("beta", 1., "Value loss weight.")
flags.DEFINE_float("max_update_steps", int(40), "Maximum number of steps before performing an update (0 for full episode).")
flags.DEFINE_integer("max_steps", int(0), "Max steps per episode (0 for no max).")
flags.DEFINE_integer("max_global_steps", int(0), "Max global steps (0 for no max).")
flags.DEFINE_integer("num_episodes", int(1500), "Total episodes for training (0 for no max).")
flags.DEFINE_integer("num_sessions", int(1), "Total training sessions.")
flags.DEFINE_integer("snapshot_step", int(15), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_integer("num_gpus", 2, "Number of GPUs for training.")
flags.DEFINE_integer("num_cpus", 8, "Number of CPUs for training.")
flags.DEFINE_integer("parallel", 2, "How many instances to run in parallel.")
flags.DEFINE_integer("stagger", int(1), "Delay between each initial agent run.")
flags.DEFINE_bool("dual_msprop", False, "Train policy and value nets separately.")
flags.DEFINE_bool("rand_hyper", False, "Randomize hyper-parameters.")

# Scenario Flags
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 32, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 32, "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

# Agent Flags
flags.DEFINE_string("network", "networks.fully_conv.FullyConv", "Network to train on")
flags.DEFINE_string("agent", "agents.a3c.agent.A3CAgent", "Agent to use")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(), "Bot's strength.")

# Meta Flags
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")
flags.DEFINE_string("save_dir", "replays/", "Directory where replays will be saved")
flags.DEFINE_integer("save_replay_frequency", 1000, "Frequency to save a replay")

config = flags.FLAGS
config(sys.argv)

# Set up devices for multi-threading
if config.training:
    PARALLEL_COUNT = config.parallel
    DEVICE_GPU = ['/gpu:' + str(gpu) for gpu in range(config.num_gpus)]
    DEVICE_CPU = ['/cpu:' + str(cpu) for cpu in range(config.num_cpus)]
else:
    PARALLEL_COUNT = 1
    DEVICE = ['/cpu:0']


def _main(unused_argv):
    snapshot_path = config.snapshot_path + config.map + '/' + config.agent + '/' + config.network
    for i in range(config.num_sessions):
        # Set up directory folders for logs and checkpoints
        log = config.log_path + config.map + '/' + config.agent + '/' + config.network + '/' + str(time.time())
        config.snapshot_path = snapshot_path + '_' + str(i)
        if not os.path.exists(log):
            os.makedirs(log)
        if not os.path.exists(config.snapshot_path):
            os.makedirs(config.snapshot_path)

        if config.rand_hyper:
            config.learning_rate = np.random.uniform(0.0, 1.0) * 10**np.random.uniform(-5.0, -2.0, 1)[0]
            config.beta = np.random.uniform(0.0, 1.0)
            config.eta = np.random.uniform(0.0, 0.1)
        maps.get(config.map)  # Assert that map exists.

        # Configure session
        tf.reset_default_graph()
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        session = tf.Session(config=sess_config)
        summary_writer = tf.summary.FileWriter(log)

        # Parse agent class
        agent_module, agent_name = config.agent.rsplit(".", 1)
        agent_cls = getattr(importlib.import_module(agent_module), agent_name)

        agent = agent_cls(config,
                          session,
                          summary_writer,
                          config.snapshot_path)

        # Initialize agent and run
        agent.initialize(config, DEVICE_GPU, PARALLEL_COUNT)
        agent.run(config)


if __name__ == "__main__":
    app.run(_main)
