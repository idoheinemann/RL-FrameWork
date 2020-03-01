import abc

from rl_base.agent.agent import Agent


class HumanAgent(Agent, abc.ABC):

    def conclude(self):
        self.memory.clear()
