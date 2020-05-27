import numpy as np

from example.snake.snake_environment import SnakeEnvironment
from rl_base.agent.agent import Agent
from rl_base.agent.board_game_agent import BoardGameAgent
from rl_base.agent.video_game_agent import VideoGameAgent
from rl_base.ml.neural_network import NeuralNetwork
from rl_base.ml.tools.functions import Tanh, Linear, Relu, SoftMax, LinearRelu, Sigmoid


class SnakeAgent(VideoGameAgent):
    def __init__(self):
        super().__init__(NeuralNetwork((11, 64, 64, 3), (Sigmoid, Sigmoid, SoftMax), alpha=0.1,
                                       n_min=-0.5, n_max=0.5,
                                       gradient_method=None), 3, max_memory_size=500, epsilon_decay=0.999)
