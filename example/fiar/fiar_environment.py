import numpy as np

from rl_base.agent.agent import Agent
from rl_base.env.board_game_environment import BoardGameEnvironment


class FIAREnvironment(BoardGameEnvironment):
    BOARD_WIDTH = 7
    BOARD_HEIGHT = 6

    def __init__(self, first_player, second_player):
        super().__init__(first_player, second_player)
        self.board = np.zeros((self.BOARD_WIDTH, self.BOARD_HEIGHT, 3))
        self.board[:, :, 0] = 1.0
        self.last_in_row = [0] * self.BOARD_WIDTH

    def get_state(self, agent: Agent):
        return self.board.flatten()

    def _perform(self, index, agent):
        if self.last_in_row[index] < self.BOARD_HEIGHT:
            self.board[index, self.last_in_row[index]] = [0, 0, 0]
            self.board[index, self.last_in_row[index]][self._get_agent_index(agent)] = 1
            self.last_in_row[index] += 1
            return True
        return False

    def _is_game_over(self):
        for x in range(self.BOARD_WIDTH):
            for y in range(self.BOARD_HEIGHT):
                can_cross = 0
                if x <= self.BOARD_WIDTH - 4:
                    can_cross += 1
                    if self._are_points_win(self.board[x, y], self.board[x + 1, y], self.board[x + 2, y],
                                            self.board[x + 3, y]):
                        return True
                if y <= self.BOARD_HEIGHT - 4:
                    if self._are_points_win(self.board[x, y], self.board[x, y + 1], self.board[x, y + 2],
                                            self.board[x, y + 3]):
                        return True
                    can_cross += 1
                if can_cross == 2:
                    if self._are_points_win(self.board[x, y], self.board[x + 1, y + 1], self.board[x + 2, y + 2],
                                            self.board[x + 3, y + 3]):
                        return True
        return not any(self.board.flatten()[::3] == 0)

    def _are_points_win(self, p1, p2, p3, p4):
        if p1[0] == 0 and all(p1 == p2) and all(p1 == p3) and all(p1 == p4):
            self.winner = self.first_player if p1[1] == 1 else self.second_player
            return True
        return False

    def get_player_array(self, idx):
        return self.board.flatten()[idx::3].reshape(self.BOARD_WIDTH, self.BOARD_HEIGHT)

    def __str__(self):
        return str((self.get_player_array(1) + 2 * self.get_player_array(2)).astype(int).T[::-1]) \
            .replace('1', 'R') \
            .replace('2', 'Y') \
            .replace('0', '-')
