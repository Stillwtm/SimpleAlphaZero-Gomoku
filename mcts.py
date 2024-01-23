import torch
import numpy as np
import copy

class Node:
    """Monte-Carlo tree node.
    """
    def __init__(self, prior, parent=None):
        self._parent = parent
        self._children = {}
        
        self._W = 0  # sum of values
        self._N = 0  # number of visits
        self._P = prior  # prior probability

    def select(self, c_puct):
        """Select action among children according to value Q+u.

        Returns:
            tuple: (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1]._uct_plus_q(c_puct))

    def expand(self, actions, priors):
        """Expand tree by creating new children.

        Args:
            action_priors (list): A list of (available actions, prior probability).
        """
        for action, prob in zip(actions, priors):
            if action not in self._children:
                self._children[action] = Node(prob, self)

    def update(self, value):
        """Recursively update tree nodes.

        Args:
            value (float): Current state evaluation, from current player's perspective.
        """
        if self._parent:
            self._parent.update(-value)

        self._N += 1
        self._W += value

    def _uct_plus_q(self, c_puct):
        """Calculate upper confidence bound for trees(UCT) plus Q value.

        Args:
            c_puct (float): controlling the relative impact of value Q, 
                            and prior probability P.
        """
        u = c_puct * self._P * np.sqrt(self._parent._N) / (1 + self._N)
        Q = self._W / self._N if self._N else 0
        return Q + u
    
    def is_leaf(self):
        """Whether this node is a leaf.
        """
        return self._children == {}
    
class MCTS:
    """Monte-Carlo Tree Search.
    """
    def __init__(self, model, c_puct, first_n_moves, n_playout, coef_noise):
        self._root = Node(1.0, None)
        self._model = model
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._first_n_moves = first_n_moves
        self._coef_noise = coef_noise

    def sample_action(self, state, temp):
        """Choose action according to probabilities (proportional to N^(1/temp)).

        Args:
            state (GameState)
            temp (float): Temperature parameter for exploration.

        Returns:
            tuple: Tuple of actions and corresponding probabilities.
        """
        available_moves = state.get_empty_pos()
        assert len(available_moves) > 0

        if len(state.get_filled_pos()) < self._first_n_moves:
            temp = 1

        move_probs = np.zeros(state.chessboard.size)
        actions, probs = self._get_action_probs(state, temp)
        move_probs[actions] = probs

        # Add Dirichlet Noise for exploration
        move = np.random.choice(
            actions, p = 0.75 * probs + 0.25 * np.random.dirichlet(self._coef_noise * np.ones(len(probs))))
        
        self._update_root(move)

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
        
        self._update_root(-1)

        return move

    def _get_action_probs(self, state, temp):
        """Get action probabilities, proportional to N^(1/temp).

        Args:
            state (GameState)
            temp (float): Temperature parameter for exploration.

        Returns:
            tuple: Tuple of all actions and corresponding probabilities.
        """
        for _ in range(self._n_playout):
            self._playout(copy.deepcopy(state))
        
        # act_visits = [(act, node._N) for act, node in self._root._children.keys()]
        # acts, n_visits = zip(*act_visits)
        acts = list(self._root._children.keys())
        n_visits = [node._N for node in self._root._children.values()]
        act_probs = self._softmax(np.log(np.array(n_visits) + 1e-10) / temp)

        return acts, act_probs

    def _playout(self, state):
        """Play from current state to a leaf node.

        Args:
            state (GameState)
        """
        node = self._root
        while not node.is_leaf():
            # Greedily select next move
            action, node = node.select(self._c_puct)
            state.transition(action)

        # value = state.get_reward()
        # if value is None:
        #     probs, value = self._model(
        #         torch.FloatTensor(state.get_format_state())[None])
        #     # TODO: Maybe put all operations to gpu?
        #     probs = probs.detach().squeeze().cpu().numpy()
        #     value = value.detach().squeeze().cpu().numpy()
        #     avail_actions = state.get_empty_pos()
        #     node.expand(avail_actions, probs[avail_actions])
        
        end, winner = state.terminated()
        if end:
            if winner == 0:
                value = 0.
            else:
                value = 1. if winner == state.player_color else -1.
        else:
            probs, value = self._model(
                torch.FloatTensor(state.get_format_state())[None])
            probs = probs.detach().squeeze().cpu().numpy()
            value = value.detach().squeeze().cpu().numpy()
            avail_actions = state.get_empty_pos()
            node.expand(avail_actions, probs[avail_actions])

        node.update(-value)

        # TODO: check correctness
        # if not end:
        #     node.expand(action_probs)
        # else:
        #     # for end state，return the "true" leaf_value
        #     if winner == -1:  # tie
        #         leaf_value = 0.0
        #     else:
        #         leaf_value = (
        #             1.0 if winner == state.get_current_player() else -1.0
        #         )

        # Update value and visit count of nodes in this traversal.
        

    def _update_root(self, last_move):
        """Update the root node and reuse the search tree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = Node(1.0, None)

    def reset_root(self):
        self._root = Node(1.0, None)        

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0)
    

