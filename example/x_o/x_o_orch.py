import os
import sys
import pickle as pkl

from example.x_o.x_o_agent import XOAgent
from example.x_o.x_o_environment import XOEnvironment
from example.x_o.x_o_human import XOHuman


def game(x_player, o_player):
    env = XOEnvironment(x_player=x_player, o_player=o_player)
    has_human_players = False
    if isinstance(x_player, XOHuman):
        has_human_players = True
        x_player.env = env
    if isinstance(o_player, XOHuman):
        has_human_players = True
        o_player.env = env
    while True:
        env.cycle(x_player)
        if not env.should_continue:
            break
        env.cycle(o_player)
        if not env.should_continue:
            break
    x_player.conclude()
    o_player.conclude()
    if has_human_players:
        print(env.get_player_array(1) + 2*env.get_player_array(2))
    return str(env.winner)


def main():
    if os.path.exists('x_player.pkl'):
        x_player = pkl.load(open('x_player.pkl', 'rb'))
    else:
        x_player = XOAgent('X')
    if os.path.exists('o_player.pkl'):
        o_player = pkl.load(open('o_player.pkl', 'rb'))
    else:
        o_player = XOAgent('O')
    num_games = 10000000
    for i in range(num_games):
        game(x_player, o_player)
        sys.stdout.write(f'\rgames completed: {i / num_games * 100}%')
    pkl.dump(x_player, open('x_player.pkl', 'wb'))
    pkl.dump(o_player, open('o_player.pkl', 'wb'))
    print(x_player.net.layers[0].max())
    x_player.epsilon = 0
    o_player.epsilon = 0
    while True:
        new_x_player = XOHuman(None)
        print(game(new_x_player, o_player))

        new_o_player = XOHuman(None)
        print(game(x_player, new_o_player))


if __name__ == '__main__':
    main()
