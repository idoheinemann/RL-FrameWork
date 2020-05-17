import os
import sys
import pickle as pkl
import time

import matplotlib.pyplot as plt

from example.prisoner_dilemma.prisoner_dilemma_agent import PrisonerDilemmaAgent
from example.prisoner_dilemma.prisoner_dilemma_environment import PrisonerDilemmaEnvironment
from module_tools.string_tools import seconds_to_string

USE_PICKLED = False


def main():
    if os.path.exists('first_player.pkl') and USE_PICKLED:
        first_player = pkl.load(open('first_player.pkl', 'rb'))
        first_player.epsilon = 0.1
        print('using pickled first')
    else:
        first_player = PrisonerDilemmaAgent('RED')
    if os.path.exists('second_player.pkl') and USE_PICKLED:
        second_player = pkl.load(open('second_player.pkl', 'rb'))
        second_player.epsilon = 0.1
        print('using pickled yellow')
    else:
        second_player = PrisonerDilemmaAgent('YELLOW')
    num_games = 100000  # 10000000 to train from scratch
    results = []
    start_time = time.time()
    i = 0
    try:
        for i in range(1, num_games + 1):
            results.append(PrisonerDilemmaEnvironment.game(first_player, second_player))
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
        print('Interrupted')
    print('finished running games')
    print(first_player.epsilon)
    print(first_player.model.U[0].max())
    print(first_player.model.U[0].min())
    x = list(range(i - 1))
    plt.plot(x, [a[0] for a in results])
    plt.plot(x, [a[1] for a in results])
    plt.plot(x, [sum(a) for a in results])
    plt.show()
    if USE_PICKLED:
        pkl.dump(first_player, open('first_player.pkl', 'wb'))
        pkl.dump(second_player, open('second_player.pkl', 'wb'))

    import IPython
    IPython.embed()
    # return  # For repetitive training


if __name__ == '__main__':
    main()
