import abc
from typing import TypeVar, Tuple

from rl_base.agent.agent import Agent

Reward = TypeVar('Reward')
State = TypeVar('State')
Action = TypeVar('Action')


class Environment(abc.ABC):

    def cycle(self, agent: Agent):
        state = self.get_state(agent)
        action = agent.choose_action(state)
        reward, action = self.perform(agent, action)
        new_state = self.get_state(agent)
        agent.learn(state=state, action=action, reward=reward, new_state=new_state)

    @abc.abstractmethod
    def get_state(self, agent: Agent) -> State:
        pass

    @abc.abstractmethod
    def perform(self, agent: Agent, action) -> Tuple[Reward, Action]:
        pass
