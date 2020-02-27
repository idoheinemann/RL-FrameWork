from rl_base.agent.agent import Agent


class XOAgent(Agent):

    def __init__(self):
        super().__init__()
        self.net = None

    def choose_action(self, state):
        pass

    def conclude(self):
        pass
