# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

# from __future__ import print_function
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet


class Human(object):
    """human player"""

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 8, 8
    model_file = 'current_policy.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # 创建AI player
        best_policy = PolicyValueNet(width, height,
                                     model_file=model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5, n_playout=400)

        # 创建Human player, 输入样例: 2,3
        human = Human()

        # 设置start_player=0可以让人类先手
        game.start_play(human, mcts_player,
                        start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
