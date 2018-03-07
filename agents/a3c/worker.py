from pysc2.agents.base_agent import BaseAgent

class Worker(BaseAgent):
    def __init__(self, device, session, s_size, a_size, name):
        super().__init__()
        self.name = "worker_" + str(name)
        self.id = name

    def build_model(self, reuse, device):
        pass

    def initialize(self):
        pass

    def reset(self):
        pass

    def step(self, obs):
        pass

    def update(self, replay_buffer, discount, learning_rate, counter):
        pass

    def save_model(self, path, counter):
        pass

    def load_model(self, path):
        pass
