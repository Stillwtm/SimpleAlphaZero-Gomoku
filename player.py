import torch
import numpy as np
from alphazero.mcts import MCTS
from alphazero.model import PolicyValueNet
from minimax.minimax import MinimaxPlayer
from minimax.game_state import GameState as MinimaxGameState

class AlphaZeroPlayer:
    """A simple wrapper to perform AlphaZero move.
    """
    def __init__(self, cfg):
        self.model = PolicyValueNet(
            cfg.board_size, cfg.history, cfg.lr, cfg.num_layers, device=cfg.device)
        self.mcts = MCTS(self.model, cfg.c_puct, cfg.first_n_moves, cfg.n_playout, cfg.coef_noise)
    
    def move(self, state):
        """Main interface to make next movement.
        """
        action = self.mcts.get_action(state)

        return state.transform_action(action)
        
    def load_model(self, model_path):
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt)

class MinimaxPlayerWrapper:
    """Wrap a MinimaxPlayer to fit MCTS-style interface.
    """
    def reset_root(self):
        pass

    def get_action(self, state):
        """Transform a AlphaZero state to a Minimax state and get 1-dim action. 
        """
        last_drop = -1 if state.last_drop is None else state.last_drop
        minimax_state = MinimaxGameState(state.chessboard, state.player_color, last_drop)
        if len(state.get_filled_pos()) == 0:
            action = (state.chessboard.shape[0] // 2, state.chessboard.shape[1] // 2) 
        else:
            action = MinimaxPlayer().move(minimax_state)
        return state.inverse_action(action)

