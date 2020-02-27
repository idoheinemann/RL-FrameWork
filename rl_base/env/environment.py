import abc
from typing import TypeVar

from rl_base.agent.agent import Agent

Reward = TypeVar('Reward')
State = TypeVar('State')


class Environment(abc.ABC):

    def cycle(self, agent: Agent):
        state = self.get_state(agent)
        action = agent.choose_action(state)
        reward = self.perform(agent, action)
        agent.learn(state=state, action=action, reward=reward)

    @abc.abstractmethod
    def get_state(self, agent: Agent) -> State:
        pass

    @abc.abstractmethod
    def perform(self, agent: Agent, action) -> Reward:
        pass
