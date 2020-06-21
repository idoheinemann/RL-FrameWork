import numpy as np

from rl_base.agent.agent import Agent


class FIARHuman(Agent):
    def __init__(self):
        super().__init__()

    def choose_action(self, state):
        while True:
            try:
                x = int(input('Enter Row Index >>> '))
                action = np.zeros(7)
                action[x] = 1
                break
            except Exception as e:
                print(f'Illegal state detected: {e}')
        return action

    def conclude(self):
        pass

    def __str__(self):
        return 'HUMAN'
