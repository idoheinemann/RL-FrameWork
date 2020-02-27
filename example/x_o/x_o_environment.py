from typing import Tuple

from rl_base.agent.agent import Agent
from rl_base.env.environment import Environment

import numpy as np


class XOReward:
    def __init__(self, index, env, agent):
        self.index = index
        self.env = env
        self.agent = agent

    def get(self) -> float:
        return self.env.rewards[self.agent][self.index]


class XOEnvironment(Environment):
    def __init__(self, x_player, o_player):
        self.board = np.zeros((3, 3, 3))
        self.board[:, :, 0] = 1.0
        self.x_player = x_player
        self.o_player = o_player
        self.winner = None
        self.rewards = {x_player: [], o_player: []}

    def get_state(self, agent: Agent):
        return self.board.flatten()

    def perform(self, agent: Agent, action: np.ndarray):
        action = action.flatten()
        action = sorted(enumerate(action), key=lambda x: x[1], reverse=True)
        for i, p in action:
            if self._perform(i, agent):
                break
        self.rewards[agent].append(0)
        if self.is_game_over():
            self.rewards[agent][-1] = self._get_reward(agent)
            self.rewards[self._get_other_agent(agent)][-1] = self._get_reward(self._get_other_agent(agent))
        return XOReward(index=len(self.rewards[agent]) - 1, env=self, agent=agent)

    def _get_other_agent(self, agent):
        if agent is self.x_player:
            return self.o_player
        return self.x_player

    @staticmethod
    def _get_board_index(number: int) -> Tuple[int, int]:
        return number // 3, number % 3

    def _get_agent_index(self, agent):
        return int(agent is self.o_player) + 1

    def _perform(self, index, agent):
        index = self._get_board_index(index)
        loc = self.board[index]
        if loc[0]:
            self.board[index] = [0, 0, 0]
            self.board[index][self._get_agent_index(agent)] = 1
            return True
        return False

    def _get_reward(self, agent):
        if self.winner is None:
            return 0
        return 1 if agent is self.winner else -1

    def is_game_over(self):
        for i in range(3):
            point = self.board[i, 0]
            if all(self.board[i, 1] == point) and all(self.board[i, 2] == point) and point[0] != 0:
                self.winner = self.x_player if point[1] == 1 else self.o_player

        for i in range(3):
            point = self.board[0, i]
            if all(self.board[1, i] == point) and all(self.board[2, i] == point) and point[0] != 0:
                self.winner = self.x_player if point[1] == 1 else self.o_player

        point = self.board[0, 0]
        if all(self.board[1, 1] == point) and all(self.board[2, 2] == point) and point[0] != 0:
            self.winner = self.x_player if point[1] == 1 else self.o_player

        point = self.board[2, 0]
        if all(self.board[1, 1] == point) and all(self.board[0, 2] == point) and point[0] != 0:
            self.winner = self.x_player if point[1] == 1 else self.o_player

        return self.winner is not None
