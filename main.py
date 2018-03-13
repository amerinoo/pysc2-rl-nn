import importlib
import os
import sys

from pysc2 import maps
from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import stopwatch
import tensorflow as tf

from absl import app
from absl import flags

import time

FLAGS = flags.FLAGS

# Training Flags
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_float("max_update_steps", 0, "Maximum number of steps before performing an update")
flags.DEFINE_integer("max_steps", int(0), "Max steps per episode.")
flags.DEFINE_integer("max_global_steps", int(0), "Max global steps.")
flags.DEFINE_integer("num_episodes", int(1000), "Total episodes for training")
flags.DEFINE_integer("snapshot_step", int(15), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_integer("num_gpus", 2, "Number of GPUs for training")
flags.DEFINE_integer("num_cpus", 8, "Number of CPUs for training")
flags.DEFINE_integer("parallel", 2, "How many instances to run in parallel.")

# Scenario Flags
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
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
flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")
flags.DEFINE_string("save_dir", "replays/", "Directory where replays will be saved")
flags.DEFINE_integer("save_replay_frequency", 1000, "Frequency to save a replay")

FLAGS(sys.argv)

# Set up devices for multi-threading
if FLAGS.training:
    PARALLEL_COUNT = FLAGS.parallel
    DEVICE_GPU = ['/gpu:' + str(gpu) for gpu in range(FLAGS.num_gpus)]
    DEVICE_CPU = ['/cpu:' + str(cpu) for cpu in range(FLAGS.num_cpus)]
else:
    PARALLEL_COUNT = 1
    DEVICE = ['/cpu:0']

# Set up directory folders for logs and checkpoints
LOG = FLAGS.log_path + FLAGS.map + '/' + FLAGS.agent + '/' + str(time.time())
SNAPSHOT_PATH = FLAGS.snapshot_path + FLAGS.map + '/' + FLAGS.agent
if not os.path.exists(LOG):
    os.makedirs(LOG)
if not os.path.exists(SNAPSHOT_PATH):
    os.makedirs(SNAPSHOT_PATH)


def _main(unused_argv):
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    maps.get(FLAGS.map)  # Assert that map exists.

    # Configure session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    summary_writer = tf.summary.FileWriter(LOG)

    # Parse agent class
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)

    agent = agent_cls(FLAGS.training,
                      FLAGS.minimap_resolution,
                      FLAGS.screen_resolution,
                      FLAGS.network,
                      FLAGS.map,
                      session,
                      summary_writer)

    # Initialize agent and run
    agent.initialize(DEVICE_CPU, PARALLEL_COUNT)
    agent.run()

    # # if not FLAGS.training or FLAGS.continuation:
    # #     global COUNTER
    # #     COUNTER = agent.load_model(SNAPSHOT_PATH)

    if FLAGS.profile:
        print(stopwatch.sw)


if __name__ == "__main__":
    app.run(_main)