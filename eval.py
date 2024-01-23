import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from collections import defaultdict
import argparse

from mcts import MCTS
from game import Game
from config import Config
from model import PolicyValueNet
from player import MinimaxPlayerWrapper

class Workspace:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def get_player(self, p, model_path=None):
        if p == 'alphazero':
            model = PolicyValueNet(
                self.cfg.board_size, self.cfg.history, self.cfg.lr, self.cfg.num_layers, device=self.cfg.device)
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt)
            return MCTS(model, self.cfg.c_puct, self.cfg.first_n_moves, self.cfg.n_playout, self.cfg.coef_noise)
        elif p == 'minimax':
            return MinimaxPlayerWrapper()
        else:
            raise ValueError('Unsupport player type!')

    def policy_evaluate(self, p1, p2, n_games=10, model_path=None, render=False):
        """Evaluate the trained policy by playing against current best player.
        """
        player1 = self.get_player(p1, model_path)
        player2 = self.get_player(p2, model_path)
        game = Game(self.cfg.board_size, self.cfg.history)
        
        win_cnt = defaultdict(int)
        for i in range(n_games):
            if i % 2 == 0:
                winner = game.play(player1, player2, verbose=True, render=render, random_start=True)
                win_cnt[winner] += 1
            else:
                winner = game.play(player2, player1, verbose=True, render=render, random_start=True)
                win_cnt[-winner] += 1

            # import ipdb; ipdb.set_trace()

        win_ratio = (win_cnt[-1] + 0.5 * win_cnt[0]) / n_games
        
        print(f"Eval | Total: {n_games} | Win: {win_cnt[-1]} | Lose: {win_cnt[1]} | Tie: {win_cnt[0]} | Win ratio: {win_ratio}")
        
        return win_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1', type=str, default='alphazero', choices=['minimax', 'alphazero'])
    parser.add_argument('--p2', type=str, default='alphazero', choices=['minimax', 'alphazero'])
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--model_path', type=str, default='./ckpt8x8.pt')
    parser.add_argument('--render', action='store_true', default=False)
    args = parser.parse_args()

    cfg = Config()
    w = Workspace(cfg)
    w.policy_evaluate(args.p1, args.p2, n_games=args.games, model_path=args.model_path, render=args.render)
    