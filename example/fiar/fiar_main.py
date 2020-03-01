import os
import sys
import pickle as pkl
import time

from example.fiar.fiar_agent import FIARAgent
from example.fiar.fiar_environment import FIAREnvironment
from example.fiar.fiar_human import FIARHuman
from tools.string_tools import seconds_to_string


def main():
    if os.path.exists('red_player.pkl'):
        red_player = pkl.load(open('red_player.pkl', 'rb'))
        red_player.epsilon = 0.1
    else:
        red_player = FIARAgent('RED')
    if os.path.exists('yellow_player.pkl'):
        yellow_player = pkl.load(open('yellow_player.pkl', 'rb'))
        yellow_player.epsilon = 0.1
    else:
        yellow_player = FIARAgent('YELLOW')
    num_games = 10000000  # 10000000 to train from scratch
    start_time = time.time()
    for i in range(num_games):
        FIAREnvironment.game(red_player, yellow_player)
        completed = i / num_games
        time_diff = time.time() - start_time
        time_left = time_diff / (completed + 1e-100) - time_diff
        percent_string = str(completed * 100)[:6]
        if len(percent_string) < 6:
            percent_string += '0' * (6-len(percent_string))
        sys.stdout.write(f'\rgames completed: {percent_string}%, estimated time: {seconds_to_string(time_left)}')
    print()
    print('finished running games')
    print(red_player.model.layers[0].max())
    pkl.dump(red_player, open('red_player.pkl', 'wb'))
    pkl.dump(yellow_player, open('yellow_player.pkl', 'wb'))
    red_player.epsilon = 0
    yellow_player.epsilon = 0
    while True:
        new_x_player = FIARHuman()
        print(FIAREnvironment.game(new_x_player, yellow_player, has_human_players=True))

        new_o_player = FIARHuman()
        print(FIAREnvironment.game(red_player, new_o_player, has_human_players=True))


if __name__ == '__main__':
    main()
