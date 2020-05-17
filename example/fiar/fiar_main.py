import os
import sys
import pickle as pkl
import time

from example.fiar.fiar_agent import FIARAgent
from example.fiar.fiar_environment import FIAREnvironment
from example.fiar.fiar_human import FIARHuman
from module_tools.string_tools import seconds_to_string, delete_last_line


def main():
    if os.path.exists('red_player.pkl'):
        red_player = pkl.load(open('red_player.pkl', 'rb'))
        print('using pickled red')
    else:
        red_player = FIARAgent('RED')
    if os.path.exists('yellow_player.pkl'):
        yellow_player = pkl.load(open('yellow_player.pkl', 'rb'))
        print('using pickled yellow')
    else:
        yellow_player = FIARAgent('YELLOW')
    num_games = 100000  # 10000000 to train from scratch
    start_time = time.time()
    print()  # for delete last line
    was_interrupted = False
    try:
        for i in range(num_games):
            FIAREnvironment.game(red_player, yellow_player)
            if i == 0:
                continue
            completed = i / num_games
            time_diff = time.time() - start_time
            time_left = time_diff / (completed + 1e-100) - time_diff
            percent_string = str(completed * 100)[:6]
            if len(percent_string) < 6:
                percent_string += '0' * (6 - len(percent_string))
            # delete_last_line()
            print(f'\rgames completed: {percent_string}%, estimated time: {seconds_to_string(time_left)}', end='', flush=True)
    except KeyboardInterrupt:
        was_interrupted = True
        import traceback
        traceback.print_exc()
        print("Interrupted by user")
    print()
    print('finished running games')
    red_player.memory.clear()
    yellow_player.memory.clear()
    pkl.dump(red_player, open('red_player.pkl', 'wb'))
    pkl.dump(yellow_player, open('yellow_player.pkl', 'wb'))
    if was_interrupted:
        exit(1)
    return  # For repetitive training
    red_player.epsilon = 0
    yellow_player.epsilon = 0
    while True:
        new_x_player = FIARHuman()
        print(FIAREnvironment.game(new_x_player, yellow_player, has_human_players=True))

        new_o_player = FIARHuman()
        print(FIAREnvironment.game(red_player, new_o_player, has_human_players=True))


if __name__ == '__main__':
    main()
