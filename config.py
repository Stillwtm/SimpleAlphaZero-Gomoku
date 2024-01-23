from dataclasses import dataclass

@dataclass
class Config:
    board_size = 8
    # training parameters
    num_train_iters: int = 100000
    replay_buffer_size: int = 1000000
    batch_size: int = 16
    update_times: int = 5
    self_play_times: int = 1
    lr: float = 1e-3
    device: int = 'cuda'
    save_every: int = 50
    eval_every: int = 50
    new_best_threshold: float = 0.55
    num_eval_games: int = 5
    # model parameters
    history: int = 4
    input_dim: int = 9
    num_layers: int = 4
    # MCTS parameters
    c_puct: float = 5
    n_playout: int = 200
    first_n_moves: int = 15
    tempature: float = 1e-3
    coef_noise: float = 0.04  # 10 / n^2
    # common
    ckpt_path: str = './ckpt8x8_1.pt'
    best_path: str = './best8x8_1.pt'