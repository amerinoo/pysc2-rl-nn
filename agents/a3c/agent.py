import itertools
import threading
import time

import tensorflow as tf

from pysc2.agents import base_agent
from pysc2.env import available_actions_printer, sc2_env

from common import util

from agents.a3c.worker import Worker
from agents.a3c.estimators import configure_estimators
from agents.runner import Runner


def make_env(config):
    with sc2_env.SC2Env(
            map_name=config.map,
            agent_race=config.agent_race,
            bot_race=config.bot_race,
            difficulty=config.difficulty,
            step_mul=config.step_mul,
            screen_size_px=(config.screen_resolution, config.screen_resolution),
            minimap_size_px=(config.minimap_resolution, config.minimap_resolution),
            save_replay_episodes=config.save_replay_frequency,
            replay_dir=config.save_dir,
            visualize=config.render) as env:
        return available_actions_printer.AvailableActionsPrinter(env)


class A3CAgent(base_agent.BaseAgent):
    def __init__(self, config, session, summary_writer, snapshot_path, name="a3c_agent"):
        super().__init__()
        self.name = name
        self.summary = []

        self.session = session
        self.summary_writer = summary_writer
        self.snapshot_path = snapshot_path
        self.map_name = config.map

        # Dimensions
        self.m_size = config.minimap_resolution
        self.s_size = config.screen_resolution
        # Make sure that screen resolution is equal to minimap resolution
        assert (self.m_size == self.s_size)

        self.network_name = config.network

        self.workers = []
        self.runners = []

        # Global iterators
        self.global_step_counter = itertools.count()
        self.global_episode_counter = itertools.count()

        self.saver = None

    def initialize(self, config, device, worker_count):
        with tf.device(device[0]):
            # Keeps track of the number of updates we've performed
            global_step = tf.Variable(0, name="global_step", trainable=False)

            # Global network
            with tf.variable_scope("global"):
                network = util.init_network(self.network_name, self.m_size, self.s_size)
                features = util.init_feature_placeholders(self.m_size, self.s_size)
                policy_net, value_net, optimizers = configure_estimators(
                    network,
                    features,
                    config.eta,
                    config.beta,
                    config.learning_rate,
                    config.dual_msprop
                )

            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=10)

            # Create worker graphs
            for worker_id in range(worker_count):
                worker_summary_writer = None
                if worker_id == 0:
                    worker_summary_writer = self.summary_writer

                worker = Worker(
                    name="{}_{}".format(self.name, worker_id),
                    device=device[(worker_id + 1) % len(device)],
                    session=self.session,
                    m_size=self.m_size,
                    s_size=self.s_size,
                    global_optimizers=optimizers,
                    network=network,
                    map_name=self.map_name,
                    learning_rate=config.learning_rate,
                    discount_factor=config.discount,
                    eta=config.eta,
                    beta=config.beta,
                    summary_writer=worker_summary_writer
                )
                self.workers.append(worker)

                runner = Runner(
                    config=config,
                    agent=worker,
                    env=make_env(config),
                    global_step_counter=self.global_step_counter,
                    global_episode_counter=self.global_episode_counter,
                    saver=self.saver
                )
                self.runners.append(runner)

    def run(self, config):
        with self.session as session:
            session.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()

            if config.continuation:
                start_episode = self.load_model(self.snapshot_path)
            else:
                start_episode = 0
            self.global_episode_counter = itertools.count(start_episode)

            # Start worker threads
            threads = []
            for runner in self.runners:
                t = threading.Thread(
                    target=lambda w_fn=runner: runner.run(coord)
                )
                threads.append(t)
                t.daemon = True
                t.start()
                time.sleep(config.stagger)

            coord.join(threads)

    def load_model(self, path):
        checkpoint = tf.train.latest_checkpoint(path)
        if checkpoint:
            self.saver.restore(self.session, checkpoint)
            return int(checkpoint.split('-')[-1])
        tf.logging.info("No previous checkpoints found!")
        return 0
