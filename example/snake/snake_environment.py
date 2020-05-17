import numpy as np

from rl_base.agent.agent import Agent
from rl_base.env.environment import Environment, Reward, State


class SnakeEnvironment(Environment):
    BOARD_WIDTH = 5
    BOARD_HEIGHT = 5

    def __init__(self):
        self.board = np.zeros((self.BOARD_WIDTH, self.BOARD_HEIGHT, 2))
        self.snake_direction = np.array([0, 1], dtype=np.int)  # -up/+down, -left/+right
        self.snake_head_index = np.array([SnakeEnvironment.BOARD_WIDTH // 2, SnakeEnvironment.BOARD_HEIGHT // 2], dtype=np.int)
        self.snake_tail_index = self.snake_head_index.copy()
        self.apple_location = None
        self.board[tuple(self.snake_head_index)][0] = 1
        self.directions = []
        self.in_game = True
        self.score = 0
        self.randomize_apple()

    def randomize_apple(self):
        while True:
            apple_loc = np.random.randint(0, self.BOARD_WIDTH), np.random.randint(0, self.BOARD_HEIGHT)
            if self.board[apple_loc].sum() == 0:
                break
        self.board[apple_loc][1] = 1
        self.apple_location = np.array(apple_loc)

    def has_obstacle(self, point):
        if -1 in point:
            return False
        try:
            return self.board[tuple(point)][0]
        except IndexError:
            return 1

    def get_state(self, agent: Agent):
        relative = self.apple_location - self.snake_head_index
        return np.array([
            self.has_obstacle(self.snake_head_index + self.snake_direction),
            self.has_obstacle(self.snake_head_index + self.__rotate_left(self.snake_direction)),
            self.has_obstacle(self.snake_head_index + self.__rotate_right(self.snake_direction)),
            float(relative[0] > 0),
            float(relative[1] > 0),
            float(relative[0] < 0),
            float(relative[1] < 0),
            float(self.snake_direction[0] == -1),
            float(self.snake_direction[0] == 1),
            float(self.snake_direction[1] == -1),
            float(self.snake_direction[1] == 1)
            # float(abs(relative).sum() == 1)
        ])

    @staticmethod
    def __rotate_left(vec):
        vec = vec[::-1]
        vec[0] = -vec[0]
        return vec

    @staticmethod
    def __rotate_right(vec):
        vec = vec[::-1]
        vec[1] = -vec[1]
        return vec

    def perform(self, agent: Agent, action):  # [turn_left stay turn_right]
        _action = action.flatten().argmax()
        if _action == 0:
            self.snake_direction = self.__rotate_left(self.snake_direction)
        elif _action == 2:
            self.snake_direction = self.__rotate_right(self.snake_direction)
        self.snake_head_index = self.snake_direction + self.snake_head_index
        _reward = 0
        try:

            if self.board[tuple(self.snake_head_index)][0] == 1:
                raise IndexError('Lost')

            if -1 in self.snake_head_index:
                raise IndexError('Lost')

            self.directions.append(self.snake_direction.copy())
            self.board[tuple(self.snake_head_index)][0] = 1

            if self.board[tuple(self.snake_head_index)][1] == 1:
                _reward = 100
                self.board[tuple(self.snake_head_index)][1] = 0
                self.score += 1
                self.randomize_apple()
            else:
                self.board[tuple(self.snake_tail_index)][0] = 0
                self.snake_tail_index += self.directions.pop(0)

        except IndexError:
            self.in_game = False
            _reward = -100
        return _reward, _action

    def __str__(self):
        tmp = self.board[:, :, 0].astype(int) + 2 * self.board[:, :, 1].astype(int)
        tmp[tuple(self.snake_head_index)] = 3
        return str(tmp)

    @classmethod
    def game(cls, agent):
        env = cls()
        while env.in_game:
            env.cycle(agent)
        return env.score
