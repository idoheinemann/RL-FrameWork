import numpy as np

from example.x_o.x_o_environment import XOEnvironment
from rl_base.agent.agent import Agent


class XOHuman(Agent):
    def __init__(self, env: XOEnvironment):
        super().__init__()
        self.env = env

    def choose_action(self, state):
        print(self.env.get_player_array(1) + 2 * self.env.get_player_array(2))
        x = int(input('Enter X loc >>> '))
        y = int(input('Enter Y loc >>> '))
        action = np.zeros(9)
        action[y * 3 + x] = 1
        return action

    def conclude(self):
        pass

    def __str__(self):
        return 'HUMAN'
