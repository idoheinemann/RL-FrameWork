import os
import sys
import pickle as pkl
import time

from example.x_o.x_o_agent import XOAgent
from example.x_o.x_o_environment import XOEnvironment
from example.x_o.x_o_human import XOHuman
from tools.string_tools import seconds_to_string


def main():
    if os.path.exists('x_player.pkl'):
        x_player = pkl.load(open('x_player.pkl', 'rb'))
        x_player.epsilon = 0.1
    else:
        x_player = XOAgent('X')
    if os.path.exists('o_player.pkl'):
        o_player = pkl.load(open('o_player.pkl', 'rb'))
        o_player.epsilon = 0.1
    else:
        o_player = XOAgent('O')
    num_games = 0  # 10000000 to train from scratch
    start_time = time.time()
    for i in range(num_games):
        XOEnvironment.game(x_player, o_player)
        completed = i / num_games
        time_diff = time.time() - start_time
        time_left = time_diff / (completed + 1e-100) - time_diff
        percent_string = str(completed * 100)[:6]
        if len(percent_string) < 6:
            percent_string += '0' * (6-len(percent_string))
        sys.stdout.write(f'\rgames completed: {percent_string}%, estimated time: {seconds_to_string(time_left)}')
        sys.stdout.flush()
    pkl.dump(x_player, open('x_player.pkl', 'wb'))
    pkl.dump(o_player, open('o_player.pkl', 'wb'))
    print(x_player.model.layers[0].max())
    x_player.epsilon = 0
    o_player.epsilon = 0
    while True:
        new_x_player = XOHuman()
        print(XOEnvironment.game(new_x_player, o_player, has_human_players=True))

        new_o_player = XOHuman()
        print(XOEnvironment.game(x_player, new_o_player, has_human_players=True))


if __name__ == '__main__':
    main()
