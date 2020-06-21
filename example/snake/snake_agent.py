import numpy as np

from rl_base.agent.video_game_agent import VideoGameAgent
from rl_base.ml.neural_network import NeuralNetwork
from rl_base.ml.tools.functions import Tanh, Linear, Relu, SoftMax, LinearRelu, Sigmoid


def normalize(x):
    return max(min(1, x), -1)


class SnakeAgent(VideoGameAgent):
    def __init__(self):
        super().__init__(NeuralNetwork((7, 64, 64, 3), (Relu, Relu, Linear), alpha=0.01,
                                       n_min=-0.1, n_max=0.1,
                                       gradient_method=np.vectorize(normalize)),
                         flat_action_space=3,
                         max_memory_size=500,
                         normalize_reward=False,
                         gamma=0.9,
                         epsilon_decay=0.999,
                         min_epsilon=0.01)
