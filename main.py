import os
import sys
import importlib
import threading

from pysc2 import maps
from pysc2.env import available_actions_printer, sc2_env
from pysc2.lib import stopwatch
import tensorflow as tf

from absl import app
from absl import flags

from run_loop import run_loop


COUNTER = 0
LOCK = threading.Lock()
FLAGS = flags.FLAGS

# Training Flags
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", .001, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", int(1e5), "Max steps per episode.")
flags.DEFINE_integer("num_episodes", int(300), "Total episodes for training")
flags.DEFINE_integer("snapshot_step", int(10), "Step for snapshot.")
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
if FLAGS.training:
    PARALLEL = FLAGS.parallel
    DEVICE_GPU = ['/gpu:' + str(gpu) for gpu in range(FLAGS.num_gpus)]
    DEVICE_CPU = ['/cpu:' + str(cpu) for cpu in range(FLAGS.num_cpus)]
else:
    PARALLEL = 1
    DEVICE = ['/cpu:0']

LOG = FLAGS.log_path + FLAGS.map + '/' + FLAGS.agent
SNAPSHOT_PATH = FLAGS.snapshot_path + FLAGS.map + '/' + FLAGS.agent
if not os.path.exists(LOG):
    os.makedirs(LOG)
if not os.path.exists(SNAPSHOT_PATH):
    os.makedirs(SNAPSHOT_PATH)


def run_thread(agent, map_name, visualize):
    with sc2_env.SC2Env(
            map_name=map_name,
            agent_race=FLAGS.agent_race,
            bot_race=FLAGS.bot_race,
            difficulty=FLAGS.difficulty,
            step_mul=FLAGS.step_mul,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            save_replay_episodes=FLAGS.save_replay_frequency,
            replay_dir=FLAGS.save_dir,
            visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)

    replay_buffer = []
    for recorder, is_done in run_loop(agent, env, FLAGS.num_episodes, FLAGS.max_steps):
        if FLAGS.training:
            replay_buffer.append(recorder)
            if is_done:
                with LOCK:
                    global COUNTER
                    COUNTER += 1
                    counter = COUNTER
                print("Counter: ", counter)
                # Learning rate schedule
                learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
                agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
                replay_buffer = []
                if counter % FLAGS.snapshot_step == 1:
                    agent.save_model(SNAPSHOT_PATH, counter)
                if counter >= FLAGS.num_episodes:
                    break
        elif is_done:
            obs = recorder[-1].observation
            score = obs["score_cumulative"][0]
            print('Your score is ' + str(score) + '!')
    if FLAGS.save_replay:
        env.save_replay(agent.name)


def _main(unused_argv):
    """Run agent"""
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    maps.get(FLAGS.map)  # Assert that map exists.

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    summary_writer = tf.summary.FileWriter(LOG)

    # Potentially multiple agents
    # Setup agent
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)

    agents = []
    for i in range(PARALLEL):
        agent = agent_cls(FLAGS.training,
                          FLAGS.minimap_resolution,
                          FLAGS.screen_resolution,
                          FLAGS.network,
                          session,
                          summary_writer)

        if FLAGS.training:
            agent.build_model(i > 0, DEVICE_GPU[i % len(DEVICE_GPU)])
        else:
            agent.build_model(i > 0, DEVICE_CPU[i % len(DEVICE_CPU)])

        agent.initialize()

        agents.append(agent)

    if not FLAGS.training or FLAGS.continuation:
        global COUNTER
        COUNTER = agent.load_model(SNAPSHOT_PATH)

    # Run threads
    threads = []
    for i in range(PARALLEL):
        t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.map, False))
        threads.append(t)
        t.daemon = True
        t.start()

    for t in threads:
        t.join()

    if FLAGS.profile:
        print(stopwatch.sw)


if __name__ == "__main__":
    app.run(_main)