import tf_agents
from tf_agents.environments import utils

from .components.vrp_env import VrpEnv


class TensorflowTrainingService:
    def __init__(self, stops, agent, learning_rate, microhub):
        self.stops = stops
        self.agent = agent
        self.learning_rate = learning_rate
        self.microhub = microhub

    def define_vrp_environment(self):
        env = VrpEnv(self.microhub, len(self.stops), self.stops)
        utils.validate_py_environment(env, episodes=5)
        test = 0
        tf_env = tf_agents.environments.tf_py_environment.TFPyEnvironment(env)
