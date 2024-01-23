import torch
import numpy as np
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import copy
import os

from mcts import MCTS
from game import Game
from model import PolicyValueNet
from buffer import ReplayBuffer
from config import Config

class Workspace:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.buffer = ReplayBuffer(cfg.replay_buffer_size)
        self.model = PolicyValueNet(
            cfg.board_size, cfg.history, cfg.lr, self.cfg.num_layers, device=cfg.device)
        self.best_model = PolicyValueNet(
            cfg.board_size, cfg.history, cfg.lr, self.cfg.num_layers, device=cfg.device)
        self.mcts = MCTS(self.model, cfg.c_puct, cfg.first_n_moves, cfg.n_playout, cfg.coef_noise)
        self.writer = SummaryWriter()
        self.eval_pool = ProcessPoolExecutor(max_workers=4)

    def policy_update(self, iter_num):
        """Update the policy-value net.
        """
        total_loss = 0
        total_entropy = 0
        for _ in range(self.cfg.update_times):
            states, probs, winners = self.buffer.sample(self.cfg.batch_size)
            loss, act_entropy = self.model.update(states, probs, winners, self.cfg.device)
            total_loss += loss
            total_entropy += act_entropy
        
        self.writer.add_scalar('train/loss', total_loss / self.cfg.update_times, iter_num)
        self.writer.add_scalar('train/entropy', total_entropy / self.cfg.update_times, iter_num)
        
        print(
            f"Train | Iter: {iter_num} | "
            f"Avg loss: {total_loss / self.cfg.update_times:.2f} | "
            f"Avg entropy: {total_entropy / self.cfg.update_times:.2f} "
        )

    def policy_against_baseline(self, model, n_games=10, render=False):
        """Evaluate the trained policy by playing against minimax player.
        """
        player1 = MCTS(model, self.cfg.c_puct, self.cfg.first_n_moves, self.cfg.n_playout, self.cfg.coef_noise)
        player2 = MCTS(self.best_model, self.cfg.c_puct, self.cfg.first_n_moves, self.cfg.n_playout, self.cfg.coef_noise)
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

    def policy_evaluate(self, model, iter_num):
        """Evaluate the trained policy by playing against current best player.
        """
        best_path = self.cfg.best_path if os.path.exists(self.cfg.best_path) else None
        best_model = self.load_model(best_path)

        player1 = MCTS(model, self.cfg.c_puct, self.cfg.first_n_moves, self.cfg.n_playout, self.cfg.coef_noise)
        player2 = MCTS(best_model, self.cfg.c_puct, self.cfg.first_n_moves, self.cfg.n_playout, self.cfg.coef_noise)
        game = Game(self.cfg.board_size, self.cfg.history)
        
        win_cnt = defaultdict(int)
        for i in range(self.cfg.num_eval_games):
            if i % 2 == 0:
                winner = game.play(player1, player2, verbose=False, render=False, random_start=True)
                win_cnt[winner] += 1
            else:
                winner = game.play(player2, player1, verbose=False, render=False, random_start=True)
                win_cnt[-winner] += 1

        win_ratio = (win_cnt[-1] + 0.5 * win_cnt[0]) / self.cfg.num_eval_games

        if win_ratio > self.cfg.new_best_threshold:
            self.save_model(model, self.cfg.best_path)

        print(f"Eval | Iter: {iter_num} | Total: {self.cfg.num_eval_games} | Win: {win_cnt[-1]} | "
              f"Lose: {win_cnt[1]} | Tie: {win_cnt[0]} | Win ratio: {win_ratio}")
        
        return win_ratio, iter_num

    def train(self):
        for i in range(self.cfg.num_train_iters):
            self.collect_selfplay_data(self.cfg.self_play_times)
            if len(self.buffer) > self.cfg.batch_size:
                self.policy_update(iter_num=i)

            if i % self.cfg.eval_every == 0:
                task = partial(self.policy_evaluate, copy.deepcopy(self.model), i)
                future = self.eval_pool.submit(task)
                future.add_done_callback(
                    lambda res: self.writer.add_scalar('eval/win_ratio', res.result()[0], res.result()[1]))
            
            if i % self.cfg.save_every == 0:
                self.save_model(self.model, self.cfg.ckpt_path)
            
        self.eval_pool.shutdown()

    def collect_selfplay_data(self, n_games=1):
        """Collect self-play data for training.
        """
        game = Game(self.cfg.board_size, self.cfg.history)
        for _ in range(n_games):
            data = game.selfplay(
                player=self.mcts, player_color=np.random.choice([1, -1]), temp=self.cfg.tempature, render=False)
            self.buffer.add_seq(data)

    def save_model(self, model, path):
        """Save the model to a file.
        """
        torch.save(model.state_dict(), path)

    def load_model(self, path=None):
        model = PolicyValueNet(
            self.cfg.board_size, self.cfg.history, self.cfg.lr, self.cfg.num_layers, device=self.cfg.device)
        if path is not None:
            model_dict = torch.load(path)
            model.load_state_dict(model_dict)
        return model

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['eval_pool']
        del self_dict['writer']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    cfg = Config()
    workspace = Workspace(cfg)
    workspace.train()