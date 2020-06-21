import numpy as np
import pygame

from rl_base.agent.agent import Agent
from rl_base.env.environment import Environment, Reward, State


class SnakeEnvironment(Environment):
    BOARD_WIDTH = 5
    BOARD_HEIGHT = 5

    WINDOW_SIZE = 500

    REWARD = 10

    def __init__(self, gui=False):
        self.board = np.zeros((self.BOARD_WIDTH, self.BOARD_HEIGHT, 2))
        self.snake_direction = np.array([0, 1], dtype=np.int)  # -up/+down, -left/+right
        self.snake_head_index = np.array([SnakeEnvironment.BOARD_WIDTH // 2, SnakeEnvironment.BOARD_HEIGHT // 2],
                                         dtype=np.int)
        self.snake_tail_index = self.snake_head_index.copy()
        self.apple_location = None
        self.board[tuple(self.snake_head_index)][0] = 1
        self.directions = []
        self.in_game = True
        self.score = 0
        self.randomize_apple()
        self.gui = gui
        if gui:
            self.window_scale = self.WINDOW_SIZE // self.BOARD_WIDTH
            pygame.init()
            self.display = pygame.display.set_mode(
                (self.window_scale * self.BOARD_WIDTH, self.window_scale * self.BOARD_HEIGHT), pygame.HWSURFACE)
            pygame.display.set_caption('Snake')

    def render(self):
        self.display.fill((0, 0, 0))
        for i in range(self.BOARD_WIDTH):
            for j in range(self.BOARD_HEIGHT):
                if self.board[i, j][0]:
                    pygame.draw.rect(self.display, (0, 255, 0), (
                        i * self.window_scale, j * self.window_scale, self.window_scale, self.window_scale))
                if self.board[i, j][1]:
                    pygame.draw.rect(self.display, (255, 0, 0), (
                        i * self.window_scale, j * self.window_scale, self.window_scale, self.window_scale))

        pygame.display.update()
        pygame.display.flip()

    def randomize_apple(self):
        while True:
            apple_loc = np.random.randint(0, self.BOARD_WIDTH), np.random.randint(0, self.BOARD_HEIGHT)
            if self.board[apple_loc].sum() == 0:
                break
        self.board[apple_loc][1] = 1
        self.apple_location = np.array(apple_loc)

    def has_obstacle(self, point):
        if -1 in point:
            return 1
        try:
            return self.board[tuple(point)][0]
        except IndexError:
            return 1

    def get_state(self, agent: Agent):
        relative = np.sign(self.apple_location - self.snake_head_index)
        return np.array([
            self.has_obstacle(self.snake_head_index + self.snake_direction),
            self.has_obstacle(self.snake_head_index + self.__rotate_left(self.snake_direction)),
            self.has_obstacle(self.snake_head_index + self.__rotate_right(self.snake_direction)),
            relative[0],
            relative[1],
            self.snake_direction[0],
            self.snake_direction[1]
        ])

    @staticmethod
    def __rotate_left(vec):
        vec = vec.copy()[::-1]
        vec[0] = -vec[0]
        return vec

    @staticmethod
    def __rotate_right(vec):
        vec = vec.copy()[::-1]
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
        if self.has_obstacle(self.snake_head_index):
            self.in_game = False
            _reward = -self.REWARD
        else:
            self.directions.append(self.snake_direction.copy())
            self.board[tuple(self.snake_head_index)][0] = 1

            if self.board[tuple(self.snake_head_index)][1] == 1:
                _reward = self.REWARD
                self.board[tuple(self.snake_head_index)][1] = 0
                self.score += 1
                self.randomize_apple()
            else:
                self.board[tuple(self.snake_tail_index)][0] = 0
                self.snake_tail_index += self.directions.pop(0)

        if self.gui:
            self.render()
        return _reward, _action

    def __str__(self):
        tmp = self.board[:, :, 0].astype(int) + 2 * self.board[:, :, 1].astype(int)
        tmp[tuple(self.snake_head_index)] = 3
        return str(tmp)

    @classmethod
    def game(cls, agent, gui=False):
        env = cls(gui=gui)
        while env.in_game:
            env.cycle(agent)
        try:
            return env.score
        finally:
            del env
