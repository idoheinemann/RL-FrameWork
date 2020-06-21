from typing import Tuple

from rl_base.agent.agent import Agent

import numpy as np

from rl_base.env.board_game_environment import BoardGameEnvironment


class XOEnvironment(BoardGameEnvironment):
    REWARD = 10

    def __init__(self, first_player, second_player):
        super().__init__(first_player, second_player)
        self.board = np.zeros((3, 3, 3))
        self.board[:, :, 0] = 1.0

    def get_state(self, agent: Agent):
        return self.board.flatten()

    @staticmethod
    def _get_board_index(number: int) -> Tuple[int, int]:
        return number % 3, number // 3

    def _perform(self, index, agent):
        index = self._get_board_index(index)
        loc = self.board[index]
        if loc[0] == 1.0:
            self.board[index] = [0, 0, 0]
            self.board[index][self._get_agent_index(agent)] = 1
            return True
        return False

    def _get_reward(self, agent):
        if self.winner is None:
            return 0
        return 1 if agent is self.winner else -1

    def _is_game_over(self):
        for i in range(3):
            if self._is_three_points_win(self.board[i, 0], self.board[i, 1], self.board[i, 2]):
                return True

        for i in range(3):
            if self._is_three_points_win(self.board[0, i], self.board[1, i], self.board[2, i]):
                return True

        if self._is_three_points_win(self.board[0, 0], self.board[1, 1], self.board[2, 2]):
            return True

        if self._is_three_points_win(self.board[2, 0], self.board[1, 1], self.board[0, 2]):
            return True

        return not any(self.board.flatten()[::3])

    def _is_three_points_win(self, p1, p2, p3):
        if all(p1 == p2) and all(p3 == p1) and p1[0] == 0:
            self.winner = self.first_player if p1[1] == 1 else self.second_player
            return True
        return False

    def get_player_array(self, idx):
        return self.board.flatten()[idx::3].reshape(3, 3)

    def __str__(self):
        return str((self.get_player_array(1) + 2 * self.get_player_array(2)).astype(int)) \
            .replace('1', 'X') \
            .replace('2', 'O') \
            .replace('0', '-')
