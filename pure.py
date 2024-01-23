import numpy as np
import torch

class PureRL:
    def __init__(self, model, coef_noise):
        self._model = model
        self._noise_beg = coef_noise
        self._noise_end = 1e-3

        self._sample_cnt = 0
        self._noise_decay = 5000

    def sample_action(self, state):
        """Choose action directly according to model output.

        Args:
            state (GameState)

        Returns:
            tuple: Tuple of actions and corresponding probabilities.
        """
        available_moves = state.get_empty_pos()
        assert len(available_moves) > 0

        self._sample_cnt += 1
        coef_noise = self._noise_end + (self._noise_beg - self._noise_end) * \
            np.exp(-self._sample_cnt / self._noise_decay)

        move_probs = np.zeros(state.chessboard.size)
        actions, probs = self._get_action_probs(state)
        move_probs[actions] = probs

        # Add Dirichlet Noise for exploration
        move = np.random.choice(
            actions, p = 0.9 * probs + 0.1 * np.random.dirichlet(coef_noise * np.ones(len(probs))))

        return move, move_probs
    
    def get_action(self, state):
        """Choose action according to probabilities deterministically.

        Args:
            state (GameState)

        Returns:
            int: action
        """
        available_moves = state.get_empty_pos()
        assert len(available_moves) > 0

        move_probs = np.zeros(state.chessboard.size)
        actions, probs = self._get_action_probs(state, temp=1e-3)
        move_probs[actions] = probs

        move = np.argmax(move_probs)

        return move

    def _get_action_probs(self, state):
        """Get action probabilities (model output).
        """
        probs, _ = self._model(
                torch.FloatTensor(state.get_format_state())[None])
        return probs
    

