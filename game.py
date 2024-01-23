from state import GameState
import numpy as np

class Game:
    def __init__(self, board_size, history):
        self._board_size = board_size
        self._history = history
    
    def selfplay(self, player, player_color, temp, render=False):
        """Rollout a self-play game and store game data.

        Args:
            player (MCTS)
            player_color (int): Player color (+1/-1).
            temp (float): Temperature parameter for exploration.

        Returns:
            list: List of (board, action_probs, winner).
        """
        player.reset_root()
        state = GameState.from_empty(self._board_size, player_color, self._history)
        
        boards, mcts_probs, cur_players = [], [], []
        while True:
            action, action_probs = player.sample_action(state, temp=temp)

            boards.append(state.get_format_state())
            mcts_probs.append(action_probs)
            cur_players.append(state.player_color)

            state.transition(action)

            if render:
                self.graphic(state.chessboard, player1=player_color, player2=-player_color)

            end, winner = state.terminated()
            if end:
                # winner from the perspective of the current player of each state
                winners = np.zeros(len(cur_players))
                if winner != 0:
                    winners[np.array(cur_players) == winner] = 1.0
                    winners[np.array(cur_players) != winner] = -1.0

                return zip(boards, mcts_probs, winners)

            # value = state.get_reward()
            # if value is not None:
            #     if value == 0:
            #         winners = np.zeros(len(boards))
            #     elif value == -1:
            #         winners = np.ones(len(boards))
            #         winners[-2::-2] = -1

            #     # TEST:
            #     winners_z = np.zeros(len(cur_players))
            #     if winner != 0:
            #         winners_z[np.array(cur_players) == winner] = 1.0
            #         winners_z[np.array(cur_players) != winner] = -1.0

            #     if not np.all(winners == winners_z):
            #         print("winners", winners)
            #         print("winners_z", winners_z)
            #         print(winners == winners_z, np.all(winners == winners_z))
            #     #     assert False

            #     return zip(boards, mcts_probs, winners)
    
    def play(self, player1, player2, verbose=False, render=False, random_start=False):
        """Start a two-player game.

        Args:
            player1 (MCTS | MinimaxPlayerWrapper): First to play, +1.
            player2 (MCTS | MinimaxPlayerWrapper): Second to play, -1.

        Returns:
            int: Winner of the game.
        """
        player1.reset_root()
        player2.reset_root()
        if random_start:
            state = GameState.from_random_start(self._board_size, -1, self._history)
        else:
            state = GameState.from_empty(self._board_size, -1, self._history)
        
        while True:
            if state.player_color == -1:
                action = player1.get_action(state)
            else:
                action = player2.get_action(state)
                
            state.transition(action)
            
            if render:
                self.graphic(state.chessboard, player1=-1, player2=1)
                
            end, winner = state.terminated()
            if end:
                if verbose:
                    print("Winner:", winner)
                return winner

    def graphic(self, board, player1, player2):
        '''Draw the board and show game info.
        '''
        height, width = board.shape

        print("Player", player1, "with ●".rjust(3))
        print("Player", player2, "with ○".rjust(3))
        print(' ' * 2, end='')
        # rjust()
        for x in range(width):
            print("{0:3}".format(x), end='')
        # print('\r\n')
        print('\r')
        for i in range(height - 1, -1, -1):
            print("{0:3d}".format(i), end='')
            for j in range(width):
                p = board[i, j]
                if p == player1:
                    print('●'.center(3), end='')
                elif p == player2:
                    print('○'.center(3), end='')
                else:
                    print('-'.center(3), end='')
            # print('\r\n')
            print('\r')
        