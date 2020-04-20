import os
import sys
import pickle as pkl
import time

import matplotlib.pyplot as plt

from example.prisoner_dilemma.prisoner_dilemma_agent import PrisonerDilemmaAgent
from example.prisoner_dilemma.prisoner_dilemma_environment import PrisonerDilemmaEnvironment


def main():
    if os.path.exists('first_player.pkl'):
        first_player = pkl.load(open('first_player.pkl', 'rb'))
        first_player.epsilon = 0.1
        print('using pickled first')
    else:
        first_player = PrisonerDilemmaAgent('RED')
    if os.path.exists('second_player.pkl'):
        second_player = pkl.load(open('second_player.pkl', 'rb'))
        second_player.epsilon = 0.1
        print('using pickled yellow')
    else:
        second_player = PrisonerDilemmaAgent('YELLOW')
    num_games = 100000  # 10000000 to train from scratch
    results = []
    for i in range(num_games):
        results.append(PrisonerDilemmaEnvironment.game(first_player, second_player))
    print('finished running games')
    print(first_player.model.U[0].max())
    x = list(range(num_games))
    plt.plot(x, [a[0] for a in results])
    plt.plot(x, [a[1] for a in results])
    plt.plot(x, [sum(a) for a in results])
    plt.show()
    pkl.dump(first_player, open('first_player.pkl', 'wb'))
    pkl.dump(second_player, open('second_player.pkl', 'wb'))
    # return  # For repetitive training


if __name__ == '__main__':
    main()
