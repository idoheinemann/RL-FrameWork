from rl_base.agent.board_game_agent import BoardGameAgent
from rl_base.ml.neural_network import NeuralNetwork
from rl_base.ml.tools.functions import Tanh


class FIARAgent(BoardGameAgent):

    def __init__(self, name, gamma=0.9, epsilon=0.1):
        super().__init__(NeuralNetwork((8 * 6 * 3, 32, 8), Tanh, alpha=0.01, n_min=-0.1, n_max=0.1), 8, gamma=gamma,
                         epsilon=epsilon)
        self.name = name

    def __str__(self):
        return f'FIARAgent({self.name})'
