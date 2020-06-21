import os
import sys
import pickle as pkl
import time
import pygame
import numpy as np

from example.snake.snake_agent import SnakeAgent
from example.snake.snake_environment import SnakeEnvironment
import matplotlib.pyplot as plt
from module_tools.string_tools import seconds_to_string

USE_PICKLED = True


def main(gui=False):
    if os.path.exists('snake.pkl') and USE_PICKLED:
        try:
            snake = pkl.load(open('snake.pkl', 'rb'))
            snake.losses = []
            snake.rewards = []
            print('using pickled snake')
        except:
            import traceback
            traceback.print_exc()
            snake = SnakeAgent()
    else:
        snake = SnakeAgent()
    num_games = 10000  # 10000000 to train from scratch
    start_time = time.time()
    all_scores = []
    try:
        for i in range(1, num_games + 1):
            if np.isnan(snake.model.layers[0][0][0]):
                print('nan encountered')
                raise KeyboardInterrupt
            all_scores.append(SnakeEnvironment.game(snake, gui=gui))
            completed = i / num_games
            time_diff = time.time() - start_time
            time_left = time_diff / (completed + 1e-100) - time_diff
            percent_string = str(completed * 100)[:6]
            if len(percent_string) < 6:
                percent_string += '0' * (6 - len(percent_string))
            sys.stdout.write(f'\rgames completed: {percent_string}%, estimated time: {seconds_to_string(time_left)}')
            sys.stdout.flush()

    except KeyboardInterrupt:
        print()
        print("Interrupted")
    print()
    pygame.quit()
    rewards = snake.rewards
    losses = snake.losses
    snake.losses = []
    snake.rewards = []

    print(snake.epsilon)
    for i in snake.model.layers:
        print(i.max(), i.min())

    if USE_PICKLED:
        pkl.dump(snake, open('snake.pkl', 'wb'))
    # norm_const = 1 #len(all_scores) // 1000
    # norm = len(all_scores) // norm_const
    # all_scores = np.convolve(all_scores, np.ones(norm) / norm)
    norm = len(rewards) // 100
    rewards = np.convolve(rewards, np.ones(norm) / norm)
    norm = len(losses) // 100
    losses = np.convolve(losses, np.ones(norm) / norm)
    plt.plot(range(len(all_scores)), all_scores)
    plt.show()
    plt.plot(range(len(losses)), losses)
    plt.show()
    plt.plot(range(len(rewards)), rewards)
    plt.show()
    import IPython
    IPython.embed()


if __name__ == '__main__':
    main()
