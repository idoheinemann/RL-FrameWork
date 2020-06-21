import abc

import numpy as np

from .environment import Environment
from ..agent.agent import Agent


class BoardGameEnvironment(Environment, abc.ABC):
    REWARD = 1

    class BoardGameReward:
        def __init__(self, index, env, agent):
            self.index = index
            self.env = env
            self.agent = agent

        def get(self) -> float:
            return self.env.rewards[self.agent][self.index]

    def __init__(self, first_player, second_player):
        self.first_player = first_player
        self.second_player = second_player
        self.winner = None
        self.rewards = {first_player: [], second_player: []}
        self.should_continue = True

    def perform(self, agent: Agent, action: np.ndarray):
        action = action.flatten()
        action = sorted(enumerate(action), key=lambda x: x[1], reverse=True)
        action_index = -1
        for i, p in action:
            if self._perform(i, agent):
                action_index = i
                break
        self.rewards[agent].append(0)
        if self._is_game_over():
            self.should_continue = False
            self.rewards[agent][-1] = self._get_reward(agent)
            self.rewards[self._get_other_agent(agent)][-1] = self._get_reward(self._get_other_agent(agent))
        return self.BoardGameReward(index=len(self.rewards[agent]) - 1, env=self, agent=agent), action_index

    @abc.abstractmethod
    def _perform(self, index: int, agent: Agent) -> bool:
        pass

    def _get_agent_index(self, agent):
        return 1 if agent is self.first_player else 2

    def _get_other_agent(self, agent):
        return self.first_player if agent is self.second_player else self.second_player

    def _get_reward(self, agent):
        if self.winner is None:
            return 0
        return self.REWARD if agent is self.winner else -self.REWARD

    @abc.abstractmethod
    def _is_game_over(self):
        pass

    @classmethod
    def game(cls, first_player, second_player, env=None, has_human_players=False):
        if env is None:
            env = cls(first_player, second_player)
        while True:
            if has_human_players:
                print(env)
            env.cycle(first_player)
            if not env.should_continue:
                break
            if has_human_players:
                print(env)
            env.cycle(second_player)
            if not env.should_continue:
                break
        first_player.conclude()
        second_player.conclude()
        if has_human_players:
            print(env)
        winner = str(env.winner)
        del env
        return winner
