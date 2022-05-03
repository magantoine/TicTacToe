# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
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

    def __init__(self, epsilon=0.2, player="X", alpha=0.05, gamma=0.99):
        self.epsilon = epsilon
        self.player = player # 'X' or 'O'
        self.training = True

        # Q-values are stored separately depending on the current player
        # For each player we store the state-action pairs as a dictionary with states as key and actions as values
        # Actions are arrays of lengh 9 in which index i corresponds to playing on square i
        # The values stores in these arrays are the Q-values associated with this state and actions
        self.q_values = {
            'X': defaultdict(lambda: np.zeros(9)), # By default, each state has 9 posible actions of Q-value 0
            'O': defaultdict(lambda: np.zeros(9))
        }
        self.alpha = alpha
        self.gamma = gamma

    def set_player(self, player='X'):
        self.player = player

    def train(self):
        """Set training mode, in which random actions are chosen with probability epsilon"""
        self.training = True
    
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
    
    def act(self, grid):
        """
        chose the next move to do according to the state of the game and whether or not the agent is learning
        """
        if(self.training) :
            if(random.random() < self.epsilon): # explore new action with probability epsilon
                ## random move
                return self._random_move(grid)
            else :
                return self._best_move(grid)
        else: # if not in training, choose greedy action
            return self._best_move(grid)

    def _tuple2int(self, action):
        """Converts an action represented as a tuple to an int"""
        return int(action[0] * 3 +  action[1])

    def update(self, last_state, current_state, action, reward):
        """
        Update the Q-value associated with the last state and action performed
        according to the Q-learning update rule.
        
        last_state: grid previous to the last action performed
        current_state: grd after performing the last action
        action: last action performed
        reward: reward obtained by this player following the last action
        """
        if type(action) == tuple:
            action = self._tuple2int(action)
        state_key = self._state2key(last_state)
        prev_q_value = self.q_values[self.player][state_key][action]

        self.q_values[self.player][state_key][action] = prev_q_value + \
            self.alpha * (reward + self.gamma * self._best_move(current_state) - prev_q_value)



def play_n_games(player_1, player_2, n_games=20_000, update_players=None, verbose=False):
    """
    Play a specified number of tic tac toe games between two players.
    
    player_1: first player. Needs to implement methods `act` and `set_player`
    player_2: second player. Needs to implement methods `act` and `set_player`
    n_games: number of games played between the two players
    update_players: determines which players need to be updated after each move. Can take value 1, 2 or 'both'.
    """
    env = TictactoeEnv()
    turns = np.array(['X','O'])
    results = []
    for game_number in tqdm(range(n_games)):
        env.reset()
        grid, _, __ = env.observe()
        turns = turns[[1,0]] # Swap X and O every game
        player_1.set_player(turns[0])
        player_2.set_player(turns[1])
        
        for j in range(9):
            if env.current_player == player_1.player:
                move = player_1.act(grid)
            else:
                move = player_2.act(grid)
            
            last_state = grid
            grid, end, winner = env.step(move, print_grid=verbose)
            reward_1 = env.reward(turns[0])
            reward_2 = -1 * reward_1

            if update_players is not None:
                if update_players == 1 or update_players == 'both':
                    player_1.update(last_state, grid, move, reward_1)
                elif update_players == 2 or update_players == 'both':
                    player_2.update(last_state, grid, move, reward_2)

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


player_1 = QPlayer()
player_2 = OptimalPlayer(epsilon=0.5)
results = play_n_games(player_1, player_2, n_games=100_000, update_players=1, verbose=False)

# Group games into bins of 250 games
results['bins'] = pd.cut(results.game + 1, range(0, 100_001, 250)).apply(lambda x: x.right)

results.sample(10)


plt.figure(figsize=(20,10))
g = sns.barplot(data=results, x='bins', y='reward_1', orient='v', ci=None)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.show()

# + pycharm={"name": "#%%\n"}
# sns.lineplot?

# + pycharm={"name": "#%%\n"}
sns.lineplot(results.groupby("bins").mean(), x="game", y="reward_1")
# -

f = plt.figure(figsize=(15, 5))
g = sns.lineplot(data=results.groupby("bins").mean(), y="reward_1", x="game")
g.set_ylabel("average reward");

# + pycharm={"name": "#%%\n"}
results

# + pycharm={"name": "#%%\n"}

