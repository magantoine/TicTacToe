# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: ann
#     language: python
#     name: venv
# ---

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from tic_env import TictactoeEnv, OptimalPlayer
from collections import defaultdict
import random


# $Q(S, A) ← Q(S, A) + α(R + γ max_a Q(S′, a) − Q(S, A))$

class QPlayer:

    def __init__(self, epsilon=0.2, player="X", alpha=0.05, gamma=0.99, debug=False, seed=42):
        self.player = player # 'X' or 'O'
        self.training = True
        self.debug = debug
        self.alpha = alpha
        self.gamma = gamma
        self.last_action = None
        self.last_state = None
        self.epsilon = epsilon
        self.seed = seed

        # By default epsilon is a function of the game number n
        # For a constant epsilon we thus convert it to a constant function of n
        if type(epsilon) in [int, float]:
            self.epsilon = lambda n: epsilon


        # Q-values are stored separately depending on the current player
        # For each player we store the state-action pairs as a dictionary with states as key and actions as values
        # Actions are arrays of lengh 9 in which index i corresponds to playing on square i
        # The values stores in these arrays are the Q-values associated with this state and actions
        self.q_values = {
            'X': defaultdict(lambda: np.zeros(9)), # By default, each state has 9 posible actions of Q-value 0
            'O': defaultdict(lambda: np.zeros(9))
        }

    def set_player(self, player='X'):
        self.player = player

    def train(self):
        """Set training mode, in which random actions are chosen with probability epsilon"""
        self.training = True
    
    def toggle_debug(self):
        """Toggle debug mode, which prints Q-values after each update"""
        self.debug = not self.debug
    
    def eval(self):
        """Set evaluation mode, in which the best action is systematically chosen"""
        self.training = False
        
    def _valid_action_indices(self, grid):
        """Returns indices of allowed actions given the state of the game"""
        return np.where(grid.flatten() == 0)[0]
    
    def _invalid_action_indices(self, grid):
        """Returns indices of unallowed actions given the state of the game"""
        return np.where(grid.flatten() != 0)[0]

    def _state2key(self, grid):
        """Returns a hashable key given a 2d array"""
        return tuple(grid.flatten())

    def _random_move(self, grid):
        """Returns a random available action"""
        return int(np.random.choice(self._valid_action_indices(grid)))
        
    def _best_move(self, grid):
        """
        Given the 9-action array associated to the current grid, 
        pick the move with highest Q-value
        """
        actions = self.q_values[self.player][self._state2key(grid)]
        actions_allowed = actions.copy()
        actions_allowed[self._invalid_action_indices(grid)] = float('-inf') # set invalid actions' Q-value to -1
        return int(np.argmax(actions_allowed)) # if multiple actions have the max value, pick the first arbitrarily
    
    def act(self, grid, n=0):
        """
        chose the next move to do according to the state of the game and whether or not the agent is learning
        """
        action = None
        if(self.training) :
            if(random.random() < self.epsilon(n)): # explore new action with probability epsilon
                ## random move
                action = self._random_move(grid)
            else :
                action = self._best_move(grid)
        else: # if not in training, choose greedy action
            action = self._best_move(grid)
        
        self.last_action = action
        self.last_state = grid

        if self.debug:
            print('='*50)
            print(grid, end='\n\n')
            print(f"Player {self.player} played square {action}")
        return action

    def _tuple2int(self, action):
        """Converts an action represented as a tuple to an int"""
        return int(action[0] * 3 +  action[1])

    def update(self, current_state, reward):
        """
        Update the Q-value associated with the last state and action performed
        according to the Q-learning update rule.
        
        last_state: grid previous to the last action performed
        current_state: grd after performing the last action
        action: last action performed
        reward: total reward obtained by this player after its last action and the following opponent's action
        """
        # if type(self.action) == tuple:
        #     action = self._tuple2int(action)
            
        last_state_key = self._state2key(self.last_state)
        prev_q_value = self.q_values[self.player][last_state_key][self.last_action]

        greedy_move = self._best_move(current_state)
        greedy_q_value = self.q_values[self.player][last_state_key][greedy_move]

        self.q_values[self.player][last_state_key][self.last_action] = prev_q_value + \
            self.alpha * (reward + self.gamma * greedy_q_value - prev_q_value)
        if self.debug:
            print(f"\nGot reward {reward}")
            print(f"Previous Q-value: {prev_q_value}  \n New Q-value: {self.q_values[self.player][last_state_key][self.last_action]}", end='\n\n')
            print(f"Next best move Q-value {greedy_q_value}")


def play_n_games(player_1, player_2, n_games=20_000, update_players=None, verbose=False):
    """
    Play a specified number of tic tac toe games between two players.
    
    player_1: first player. Needs to implement methods `act` and `set_player`
    player_2: second player. Needs to implement methods `act` and `set_player`
    n_games: number of games played between the two players
    update_players: determines which players need to be updated after each move. Can take value 1, 2 or 'both'.
    """
    env = TictactoeEnv()
    turns = np.array(['O', 'X'])
    results = []
    reward_1 = reward_2 = 0
    for game_number in tqdm(range(n_games)):
        env.reset()
        grid, _, __ = env.observe()
        turns = turns[[1,0]] # Swap X and O every game
        player_1.set_player(turns[0])
        player_2.set_player(turns[1])
        
        for j in range(9):
            is_player_1_move = env.current_player == player_1.player
            if is_player_1_move:
                move = player_1.act(grid, n=game_number)
            else:
                move = player_2.act(grid, n=game_number)
            

            grid, end, winner = env.step(move, print_grid=verbose)

            last_reward_1, last_reward_2 = reward_1, reward_2
            reward_1 = env.reward(turns[0])
            reward_2 = -1 * reward_1

            # Check if we need to update any player's parameters
            if j >= 2 and update_players is not None:
                # Update player 2's params if player 1 just played 
                if end or not is_player_1_move and (update_players == 1 or update_players == 'both'):
                    player_1.update(grid, last_reward_1 + reward_1)
                    
                # Update player 2's params if player 1 just played 
                elif end or is_player_1_move and (update_players == 2 or update_players == 'both'):
                    player_2.update(grid, last_reward_2 + reward_2)

            if end:
                if verbose:
                    print('-------------------------------------------')
                    print('Game end, winner is player ' + str(winner))
                    print('Player 1 = ' +  turns[0])
                    print('Player 2 = ' +  turns[1])
                # env.render()
                # env.reset()
                results.append([game_number + 1, reward_1, reward_2, turns[0], turns[1]])
                break
    return pd.DataFrame(data=results, columns=['game', 'reward_1', 'reward_2', 'player_1', 'player_2'])


def measure_score(player_1, opponent_strategy='opt', n_games=500, seed=42):
    """
    Play a specified number of tic tac toe games between two players.
    
    player_1: player evaluated
    opponent_strategy: 'opt' or 'rand'
    n_games: number of games played between the two players
    seed: number used to determine randomness of the successive games
    """
    
    player_2 = OptimalPlayer(epsilon = 0 if opponent_strategy == 'opt' else 1)
    env = TictactoeEnv()
    turns = np.array(['O', 'X'])
    m = 0
    rng = np.random.RandomState(seed)
    for game_number in tqdm(range(n_games)):

        np.random.seed(rng.randint(low=0, high=32767))
        env.reset()
        grid, _, __ = env.observe()
        turns = turns[[1,0]] # Swap X and O every game
        player_1.set_player(turns[0])
        player_2.set_player(turns[1])
        
        for j in range(9):
            is_player_1_move = env.current_player == player_1.player
            if is_player_1_move:
                move = player_1.act(grid, n=game_number)
            else:
                move = player_2.act(grid, n=game_number)
            

            grid, end, winner = env.step(move)

            if end:
                # env.render()
                # env.reset()
                m += 1 if winner == turns[0] else -1
                break

    return m / n_games

player_1 = QPlayer(debug=False,)
player_2 = OptimalPlayer(epsilon=0.5)
results = play_n_games(player_1, player_2, n_games=20_000, update_players=1, verbose=False)
# Group games into bins of 250 games
results['bins'] = pd.cut(results.game, range(0, 20_001, 250)).apply(lambda x: x.right)
sns.lineplot(data=results.groupby("bins").mean(), x="game", y="reward_1")

# + pycharm={"name": "#%%\n"}

# -

# 2.1.1

# + pycharm={"name": "#%%\n"}
player_1 = QPlayer(epsilon = lambda n: max(0.1, 0.8 * (1 - n / 10_000)))
player_2 = OptimalPlayer(epsilon=0.5)
results = play_n_games(player_1, player_2, n_games=20_000, update_players=1, verbose=False)
# -

# Group games into bins of 250 games
results['bins'] = pd.cut(results.game, range(0, 20_001, 250)).apply(lambda x: x.right)
sns.lineplot(data=results.groupby("bins").mean(), x="game", y="reward_1")


def measure_score(player_1, opponent_strategy='opt', n_games=500, seed=42):
    """
    Play a specified number of tic tac toe games between two players.
    
    player_1: player evaluated
    opponent_strategy: 'opt' or 'rand'
    n_games: number of games played between the two players
    seed: number used to determine randomness of the successive games
    """
    
    player_2 = OptimalPlayer(epsilon = 0 if opponent_strategy == 'opt' else 1)
    env = TictactoeEnv()
    turns = np.array(['O', 'X'])
    m = 0
    rng = np.random.RandomState(seed)
    for game_number in tqdm(range(n_games)):

        np.random.seed(rng.randint(low=0, high=32767))
        env.reset()
        grid, _, __ = env.observe()
        turns = turns[[1,0]] # Swap X and O every game
        player_1.set_player(turns[0])
        player_2.set_player(turns[1])
        
        for j in range(9):
            is_player_1_move = env.current_player == player_1.player
            if is_player_1_move:
                move = player_1.act(grid, n=game_number)
            else:
                move = player_2.act(grid, n=game_number)
            

            grid, end, winner = env.step(move)

            if end:
                # env.render()
                # env.reset()
                m += 1 if winner == turns[0] else (-1 if winner == turns[1] else 0)
                break

    return m / n_games

player_1.eval()
m_rand = measure_score(player_1, 'rand', n_games=500)
m_rand

m_rand
