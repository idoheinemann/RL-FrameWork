from example.fiar.fiar_environment import FIAREnvironment
from rl_base.agent.board_game_agent import BoardGameAgent
# from rl_base.ml.keras_neural_network import KerasNeuralNetwork
from rl_base.ml.neural_network import NeuralNetwork
from rl_base.ml.tools.functions import Tanh, SoftMax


class FIARAgent(BoardGameAgent):

    def __init__(self, name):
        super().__init__(NeuralNetwork(
            (FIAREnvironment.BOARD_WIDTH * FIAREnvironment.BOARD_HEIGHT, 64, 64, FIAREnvironment.BOARD_WIDTH),
            (Tanh, Tanh, SoftMax),
            alpha=0.01), FIAREnvironment.BOARD_WIDTH, gamma=0.9, cache_size=500, normalize_reward=True,
            epsilon_decay=0.999)
        self.name = name

    def __str__(self):
        return f'FIARAgent({self.name})'
