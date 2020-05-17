from example.fiar.fiar_environment import FIAREnvironment
from rl_base.agent.board_game_agent import BoardGameAgent
# from rl_base.ml.keras_neural_network import KerasNeuralNetwork
from rl_base.ml.neural_network import NeuralNetwork
from rl_base.ml.tools.functions import Tanh


class FIARAgent(BoardGameAgent):

    def __init__(self, name, gamma=0.9, epsilon=0.2):
        super().__init__(NeuralNetwork(
            (FIAREnvironment.BOARD_WIDTH * FIAREnvironment.BOARD_HEIGHT * 3, 16, FIAREnvironment.BOARD_WIDTH), Tanh,
            alpha=1.0), FIAREnvironment.BOARD_WIDTH, gamma=gamma, epsilon=epsilon)
        self.name = name

    def __str__(self):
        return f'FIARAgent({self.name})'
