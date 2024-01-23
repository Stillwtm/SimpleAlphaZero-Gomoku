import torch
import numpy as np
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool
from functools import partial

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
            cfg.board_size, cfg.history, cfg.lr, num_layers=4, device=cfg.device)
        self.best_model = PolicyValueNet(
            cfg.board_size, cfg.history, cfg.lr, num_layers=4, device=cfg.device)
        self.mcts = MCTS(self.model, cfg.c_puct, cfg.first_n_moves, cfg.n_playout)
        self.writer = SummaryWriter()
        self.task_pool = Pool()

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

    def policy_evaluate(self, model, iter_num):
        """Evaluate the trained policy by playing against current best player.
        """
        player1 = MCTS(model, self.cfg.c_puct, self.cfg.first_n_moves, self.cfg.n_playout)
        player2 = MCTS(self.best_model, self.cfg.c_puct, self.cfg.first_n_moves, self.cfg.n_playout)
        game = Game(self.cfg.board_size, self.cfg.history)
        
        win_cnt = defaultdict(int)
        for i in range(self.cfg.num_eval_games):
            if i % 2:
                winner = game.play(player1, player2, verbose=True, render=False)
                win_cnt[winner] += 1
            else:
                winner = game.play(player2, player1, verbose=True, render=False)
                win_cnt[-winner] += 1

        win_ratio = (win_cnt[1] + 0.5 * win_cnt[0]) / self.cfg.num_eval_games
        
        self.writer.add_scalar('eval/win_ratio', win_ratio, iter_num)
        self.save_model(model, './ckpt.pt')

        print(f"Eval | Total: {self.cfg.num_eval_games} | Win: {win_cnt[1]} | Lose: {win_cnt[-1]} | Tie: {win_cnt[0]} | Win ratio: {win_ratio}")

        if win_ratio > self.cfg.new_best_threshold:
            self.best_model.load_state_dict(model.state_dict())
            self.save_model(self.best_model, './best.pt')

    def train(self):
        for i in range(self.cfg.num_train_iters):
            self.collect_selfplay_data(self.cfg.self_play_times)
            if len(self.buffer) > self.cfg.batch_size:
                self.policy_update(iter_num=i)

            if (i+1) % self.cfg.eval_every == 0:
                task = partial(self.policy_evaluate, self.model, i)
                self.task_pool.apply_async(task)

        self.task_pool.close()
        self.task_pool.join()

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

if __name__ == "__main__":
    cfg = Config()
    workspace = Workspace(cfg)
    workspace.train()