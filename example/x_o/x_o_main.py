import os
import sys
import pickle as pkl
import time

from example.x_o.x_o_agent import XOAgent
from example.x_o.x_o_environment import XOEnvironment
from example.x_o.x_o_human import XOHuman
from module_tools.string_tools import seconds_to_string

USE_PICKLED = False


def main():
    if USE_PICKLED and os.path.exists('x_player.pkl'):
        x_player = pkl.load(open('x_player.pkl', 'rb'))
        x_player.epsilon = 0.1
    else:
        x_player = XOAgent('X')
    if USE_PICKLED and os.path.exists('o_player.pkl'):
        o_player = pkl.load(open('o_player.pkl', 'rb'))
        o_player.epsilon = 0.1
    else:
        o_player = XOAgent('O')
    num_games = 1000000  # 10000000 to train from scratch
    start_time = time.time()
    try:
        for i in range(1, num_games + 1):
            XOEnvironment.game(x_player, o_player)
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
        print("interrupted")

    if USE_PICKLED:
        pkl.dump(x_player, open('x_player.pkl', 'wb'))
        pkl.dump(o_player, open('o_player.pkl', 'wb'))
    print(x_player.epsilon)
    for i in x_player.model.layers:
        print(i.max(), i.min())
    x_player.epsilon = 0
    o_player.epsilon = 0
    while True:
        new_x_player = XOHuman()
        print(XOEnvironment.game(new_x_player, o_player, has_human_players=True))

        new_o_player = XOHuman()
        print(XOEnvironment.game(x_player, new_o_player, has_human_players=True))


if __name__ == '__main__':
    main()
