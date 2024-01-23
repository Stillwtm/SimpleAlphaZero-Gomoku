import numpy as np
from collections import deque

ROW_DIR = [-1, -1, 0, 1, 1, 1, 0, -1]
COL_DIR = [0, 1, 1, 1, 0, -1, -1, -1]

class GameState:
    def __init__(self, chessboard, player_color, last_drop, history):
        self.chessboard = chessboard
        self.player_color = player_color
        self.last_drop = last_drop
        self._history = history

        self._board_history = deque(maxlen=history)
        for _ in range(history):
            self._board_history.append(np.zeros_like(self.chessboard))

    @classmethod
    def from_empty(cls, board_size, player_color, history):
        return cls(np.zeros((board_size, board_size)), player_color, None, history)

    @classmethod
    def from_random_start(cls, board_size, player_color, history, margin=2):
        """First player do a random move.
        """
        state = cls.from_empty(board_size, player_color, history)
        r = np.random.randint(margin, board_size - margin)
        c = np.random.randint(margin, board_size - margin)
        action = state.inverse_action((r, c))
        state.transition(action)
        return state

    def transition(self, action):
        """Translate to next state.

        Args:
            action (int): [0, board_size**2)

        Returns:
            GameState: Next game state.
        """
        action = self.transform_action(action)
        self.chessboard[action] = self.player_color
        self.player_color = -self.player_color
        self.last_drop = action
        self._board_history.append(self.chessboard)

    def terminated(self):
        """Whether the game is terminated (win/draw).
        """
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif len(self.get_empty_pos()) == 0:
            return True, 0
        else:
            return False, 0
        # return self.is_win(self.last_drop) or len(self.get_empty_pos()) == 0
    
    def get_format_state(self):
        """Convert state history to format of model input.
        
        Returns:
            ndarray: (1, history*2+1, n, n).
        """
        board_shape = self.chessboard.shape
        feat = np.zeros((self._history * 2 + 1, *board_shape))

        for i, board in enumerate(reversed(self._board_history)):
            feat[i][board == 1] = 1
            feat[i + self._history][board == -1] = 1
        feat[-1][:, :] = self.player_color
        
        return feat

    def transform_action(self, action):
        """Convert 1-dim action to (r, c).
        """
        r = action // self.chessboard.shape[1]
        c = action % self.chessboard.shape[1]
        return (r, c)
    
    def inverse_action(self, action):
        """Convert (r, c) action to 1-dim.
        """
        r, c = action
        return r * self.chessboard.shape[1] + c

    def get_empty_pos(self):
        """Get empty cell positions.

        Returns:
            list: List of available actions, [0, n**2).
        """
        empty_indices = np.where(self.chessboard == 0)
        return [r * self.chessboard.shape[1] + c for r, c in np.transpose(empty_indices)]

    def get_filled_pos(self):
        """Get filled cell positions.

        Returns:
            list: List of already filled positions, [0, n**2).
        """
        filled_indices = np.where(self.chessboard != 0)
        return [r * self.chessboard.shape[1] + c for r, c in np.transpose(filled_indices)]

    @staticmethod
    def is_valid_pos(r, c):
        """Whether given position is in the board.
        """
        return r >= 0 and r < 15 and c >= 0 and c < 15

    # def get_reward(self):
    #     if self.is_win():  # is_win means last player win, current player lose
    #         return -1
    #     elif len(self.get_empty_pos()) == 0:
    #         return 0
    #     else:
    #         return None

    # def is_win(self):
    #     """Whether there is a winner at current state.
    #     """
    #     if self.last_drop is None:
    #         return False
    #     return self._is_win(self.last_drop)

    # def _is_win(self, pos):
    #     """Whether there is a winner at current state.
    #     """
    #     if self._win_horizontal(pos):
    #         return True
    #     if self._win_vertical(pos):
    #         return True
    #     if self._win_main_diag(pos):
    #         return True
    #     if self._win_cont_diag(pos):
    #         return True
    #     return False

    # def _win_horizontal(self, pos):
    #     r, c = pos
    #     left = c-4 if c-4 > 0 else 0
    #     right = c+4 if c+4 < 15 else 14
    #     check = []
    #     for i in range(left, right-3):
    #         check.append(int(abs(sum(self.chessboard[r, i:i+5])) == 5))
    #     return sum(check) == 1

    # def _win_vertical(self, pos):
    #     r, c = pos
    #     top = r-4 if r-4 > 0 else 0
    #     bottom = r+4 if r+4 < 15 else 14
    #     check = []
    #     for i in range(top, bottom-3):
    #         check.append(int(abs(sum(self.chessboard[i:i+5, c])) == 5))
    #     return sum(check) == 1

    # def _win_main_diag(self, pos):
    #     r, c = pos
    #     left, top, right, bottom = 0, 0, 0, 0
    #     if r >= c:
    #         left = c-4 if c-4 > 0 else 0
    #         bottom = r+4 if r+4 < 15 else 14
    #         right = bottom - (r-c)
    #         top = left + r-c
    #     else:
    #         right = c+4 if c+4 < 15 else 14
    #         top = r-4 if r-4 > 0 else 0
    #         left = top+c-r
    #         bottom = right-(c-r)

    #     check = []
    #     if right-left > 3:
    #         for i in range(right-left-3):
    #             col = np.arange(left+i, left+i+5)
    #             row = np.arange(top+i, top+i+5)
    #             check.append(int(abs(sum(self.chessboard[row, col])) == 5))
    #     return sum(check) == 1

    # def _win_cont_diag(self, pos):
    #     r, c = pos
    #     left, top, right, bottom = 0, 0, 0, 0
    #     if r + c <= 14:
    #         top = r-4 if r-4 > 0 else 0
    #         left = c-4 if c-4 > 0 else 0
    #         bottom = r+c-left
    #         right = r+c-top
    #     else:
    #         bottom = r+4 if r+4 < 15 else 14
    #         right = c+4 if c+4 < 15 else 14
    #         top = r+c-right
    #         left = r+c-bottom

    #     check = []
    #     if right-left > 3:
    #         for i in range(right-left-3):
    #             col = np.arange(left+i, left+i+5)
    #             row = np.arange(bottom-i, bottom-i-5, -1)
    #             check.append(int(abs(sum(self.chessboard[row, col])) == 5))
    #     return sum(check) == 1
    
    def has_a_winner(self):
        '''
        judge if there's a 5-in-a-row, and which player if so
        '''
        height, width = self.chessboard.shape
        states = self.chessboard.reshape(-1)
        n = 5

        moved = self.get_filled_pos()
        # # moves have been played
        # if len(moved) < n + 2:
        #     # too few moves to get 5-in-a-row
        #     return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states[i] for i in range(m, m + n))) == 1):
                # for each move in moved moves,judge if there's a 5-in-a-row in a line
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states[i] for i in range(m, m + n * width, width))) == 1):
                # for each move in moved moves,judge if there's a 5-in-a-row in a column
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states[i] for i in range(m, m + n * (width + 1), width + 1))) == 1):
                # for each move in moved moves,judge if there's a 5-in-a-row in a top right diagonal
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states[i] for i in range(m, m + n * (width - 1), width - 1))) == 1):
                # for each move in moved moves,judge if there's a 5-in-a-row in a top left diagonal
                return True, player

        return False, 0