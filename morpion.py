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

RESULT_FOLDER ='saved_results/'


class QPlayer:

    def __init__(self, epsilon=0.2, player="X", alpha=0.05, gamma=0.99, debug=False, seed=42):
        #self.player = player # 'X' or 'O'
        self.training = True
        self.debug = debug
        self.alpha = alpha
        self.gamma = gamma

        # We store the last state and action played for each player (especially useful when learning against itself)
        self.last_action = {'X': None, 'O': None}
        self.last_state =  {'X': None, 'O': None}
        self.epsilon = epsilon
        self.seed = seed

        # By default epsilon is a function of the game number n
        # For a constant epsilon we thus convert it to a constant function of n
        if type(epsilon) in [int, float]:
            self.epsilon = lambda n: epsilon


        # We store the state-action pairs as a dictionary with states as key and actions as values
        # Actions are arrays of lengh 9 in which index i corresponds to playing on square i
        # The values stores in these arrays are the Q-values associated with this state and actions
        self.q_values = defaultdict(lambda: np.zeros(9)) # By default, each state has 9 posible actions of Q-value 0

    def train(self):
        """Set training mode, in which random actions are chosen with probability epsilon"""
        self.training = True
    
    def eval(self):
        """Set evaluation mode, in which the best action is systematically chosen"""
        self.training = False

    def toggle_debug(self):
        """Toggle debug mode, which prints Q-values after each update"""
        self.debug = not self.debug
    
    @staticmethod    
    def _valid_action_indices(grid):
        """Returns indices of allowed actions given the state of the game"""
        return np.where(grid.flatten() == 0)[0]
    
    @staticmethod
    def _invalid_action_indices(grid):
        """Returns indices of unallowed actions given the state of the game"""
        return np.where(grid.flatten() != 0)[0]

    @staticmethod
    def _state2key(grid):
        """Returns a hashable key given a 2d array"""
        return tuple(grid.flatten())

    def _random_move(self, grid):
        """Returns a random available action"""
        return int(np.random.choice(QPlayer._valid_action_indices(grid)))
        
    def _best_move(self, grid):
        """
        Given the 9-action array associated to the current grid, 
        pick the move with highest Q-value
        """
        actions_allowed = self.q_values[QPlayer._state2key(grid)].copy()
        actions_allowed[QPlayer._invalid_action_indices(grid)] = float('-inf') # set invalid actions' Q-value to -1

        # Choose the action with the highest Q-value
        # If multiple actions have the max value, pick the first one
        return int(np.argmax(actions_allowed)) 
    
    def act(self, grid, player='X', n=0):
        """
        Choose the next move according to the state of the game
        """
        action = None
        if self.training:
            if random.random() < self.epsilon(n): # explore new action with probability epsilon
                ## random move
                action = self._random_move(grid)
            else :
                action = self._best_move(grid)
        else: # if not in training, choose greedy action
            action = self._best_move(grid)
        
        # Save the current state and action for the next update
        self.last_action[player] = action
        self.last_state[player] = grid

        return action

    def update(self, current_state, reward, player):
        """
        Update the Q-value associated with the last state and action performed
        according to the Q-learning update rule.
        
        last_state: grid previous to the last action performed
        current_state: grd after performing the last action
        action: last action performed
        reward: total reward obtained by this player after its last action and the following opponent's action
        """
        
        last_state_key = QPlayer._state2key(self.last_state[player])
        last_action = self.last_action[player]
        prev_q_value = self.q_values[last_state_key][last_action]

        if reward != 0: # since the game ended, the next state's Q-value is 0
            greedy_q_value = 0
        else:
            greedy_move = self._best_move(current_state)
            current_state_key = QPlayer._state2key(current_state)
            greedy_q_value = self.q_values[current_state_key][greedy_move]

        # Update the previous state Q-value given the current state and reward observation
        self.q_values[last_state_key][last_action] = prev_q_value + \
            self.alpha * (reward + self.gamma * greedy_q_value - prev_q_value)

        if self.debug and reward != 0:
            print('='*50)
            print(self.last_state[player], end='\n\n')
            print(f"Player {player} played square {last_action}")
            
            print(f"\nGot reward {reward}")
            print(f"Previous Q-value: {prev_q_value}  \n New Q-value: {self.q_values[last_state_key][last_action]}", end='\n\n')
            print(f"Next best move Q-value {greedy_q_value}")
            input()
            
    def get_q_values(self, state):
        return self.q_values[QPlayer._state2key(state)].reshape((3,3))


def play_n_games(player_1, player_2, n_games=20_000, update_players=None, verbose=False, evaluate=None, seed=42):
    """
    Play a specified number of tic tac toe games between two players.
    
    player_1: first player. Needs to implement methods `act` and `set_player`
    player_2: second player. Needs to implement methods `act` and `set_player`
    n_games: number of games played between the two players
    update_players: determines which players need to be updated after each move. Can take value 1, 2 or 'both'.
    """
    np.random.seed(seed)
    env = TictactoeEnv()
    turns = np.array(['O', 'X'])
    results, evalutions = [], []
    reward_1 = reward_2 = 0
    rolling_win_average = 0
    for game_number in tqdm(range(n_games)):
        env.reset()
        grid, _, __ = env.observe()
        turns = turns[[1,0]] # Swap X and O every game

        for j in range(9):
            is_player_1_move = env.current_player == turns[0]
            if is_player_1_move:
                move = player_1.act(grid, player=turns[0], n=game_number)
            else:
                move = player_2.act(grid, player=turns[1], n=game_number)
            

            if env.check_valid(move):
                grid, end, winner = env.step(move, print_grid=verbose)
                reward_1 = env.reward(turns[0])
                reward_2 = -1 * reward_1
            else:
            # If a move is not valid then the current player lost the game
                end = True
                winner = turns[1] if is_player_1_move else turns[0]
                reward_1 = -1 if is_player_1_move else 1
                reward_2 = -1 * reward_1

            if verbose :
                print(f"Reward 1: {reward_1}")
                print(f"Reward 2: {reward_2}")
                
            # Check if we need to update any player's parameters
            if j >= 2 and update_players is not None:
                # Update player 1's Q-value if player 2 just played or if it's the end
                if (end or not is_player_1_move) and (update_players == 1 or update_players == 'both'):
                    if verbose :
                        print(f"Updating {player_1} with reward {reward_1}")
                    player_1.update(grid, reward_1, turns[0])
                    
                # Update player 2's Q-value if player 1 just played or if it's the end
                if (end or is_player_1_move) and (update_players == 2 or update_players == 'both'):
                    if verbose :
                        print(f"Updating {player_2} with reward {reward_2}")
                    player_2.update(grid, reward_2, turns[1])

            if end:
                if verbose:
                    print('-------------------------------------------')
                    print('Game end, winner is player ' + str(winner))
                    print('Player 1 = ' +  turns[0])
                    print('Player 2 = ' +  turns[1])
                
                rolling_win_average += reward_1
                if (game_number + 1) % 250 == 0:
                    print(rolling_win_average / 250, end='\r')
                    rolling_win_average = 0

                # Compute M_opt and M_rand on the specified players every 250 games
                if evaluate is not None and (game_number + 1) % 250 == 0:
                    evaluation = [game_number]
                    
                    if evaluate == player_1 or evaluate=="both":
                        player_1.eval()
                        m_opt = measure_score(player_1, "opt", verbose=False)
                        m_rand = measure_score(player_1, "rand", verbose=False)
                        evaluation.append(m_opt)
                        evaluation.append(m_rand)
                        player_1.train()
                    
                    # Don't evalute twice if training against itself
                    if (evaluate == player_2 or evaluate=="both") and player_1 != player_2:
                        player_2.eval()
                        m_opt = measure_score(player_2, "opt", verbose=False)
                        m_rand = measure_score(player_2, "rand", verbose=False)
                        evaluation.append(m_opt)
                        evaluation.append(m_rand)
                        player_2.train()
                        
                    evalutions.append(evaluation)
                    
                results.append([game_number + 1, reward_1, reward_2, turns[0], turns[1]])
                break
    
    game_res = pd.DataFrame(data=results, columns=['game', 'reward_1', 'reward_2', 'player_1', 'player_2'])
    perf_columns = ["game"]
    
    if evaluate == player_1 or evaluate == "both":
        perf_columns += ["player_1_opt", "player_1_rand"]
    if (evaluate == player_2 or evaluate == "both") and player_1 != player_2:
        perf_columns += ["player_2_opt", "player_2_rand"]
        
    agents_perf = pd.DataFrame(data=evalutions, columns=perf_columns)

    results = game_res if evaluate is None else game_res.merge(agents_perf, how="outer", on="game")
    return results


def measure_score(player_1, opponent_strategy='opt', n_games=500, seed=42, verbose=True):
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
    iterator = tqdm(range(n_games)) if verbose else range(n_games)
    
    if verbose:
        n_win = n_lose = n_even = 0
        
    for game_number in iterator:

        np.random.seed(rng.randint(low=0, high=32767))
        env.reset()
        grid, _, __ = env.observe()
        turns = turns[[1,0]] # Swap X and O every game
        # player_1.set_player(turns[0])
        # player_2.set_player(turns[1])
        
        for j in range(9):
            is_player_1_move = env.current_player == turns[0]
            if is_player_1_move:
                move = player_1.act(grid, player=turns[0], n=game_number)
            else:
                move = player_2.act(grid, player=turns[1], n=game_number)
            
            if env.check_valid(move):
                grid, end, winner = env.step(move, print_grid=verbose)
            else:
                # If a move is not valid then the current player lost the game
                end = True
                winner = turns[1] if is_player_1_move else turns[0]

            if end:
                gain = 1 if winner == turns[0] else (-1 if winner == turns[1] else 0)
                if verbose:
                    n_win += gain == 1
                    n_lose += gain == -1
                    n_even += gain == 0
                m += gain
                break
    if verbose:
        return m / n_games, n_win, n_lose, n_even

    return m / n_games


# ## 2.1 Learning from experts
#
# ### Question 1:

player_1 = QPlayer()
player_2 = OptimalPlayer(epsilon=0.5)
results = play_n_games(player_1, player_2, n_games=20_000, update_players=1, verbose=False)
# Group games into bins of 250 games
results['bins'] = pd.cut(results.game, range(0, 20_001, 250)).apply(lambda x: x.right)
sns.lineplot(data=results.groupby("bins").mean(), x="game", y="reward_1")


# ### Question 2

def decreasing_epsilon(n, n_star=10_000, eps_min=0.1, eps_max=0.8):
    return max(eps_min, eps_max * (1 - n / n_star))


LOAD_RESULTS = False

# + pycharm={"name": "#%%\n"}
player_1 = QPlayer(epsilon=decreasing_epsilon)
player_2 = OptimalPlayer(epsilon=0.5)
results = play_n_games(player_1, player_2, n_games=20_000, update_players=1, verbose=False)
# -

# Group games into bins of 250 games
results['bins'] = pd.cut(results.game, range(0, 20_001, 250)).apply(lambda x: x.right)
sns.lineplot(data=results.groupby("bins").mean(), x="game", y="reward_1")

# Let's try several values of $n^*$ to see its effect. We use these values of $n^*$ :

ns_star = np.arange(0,42_000,8_000)
ns_star[0] = 1
ns_star

if not LOAD_RESULTS:
    results = []

    for n_star in ns_star :
        print(f">> n* = {n_star}")
        player_1 = QPlayer(epsilon = lambda n: decreasing_epsilon(n, n_star=n_star))
        player_2 = OptimalPlayer(epsilon=0.5)
        avg_reward = play_n_games(player_1, player_2, n_games=20_000, update_players=1, verbose=False)
        avg_reward["n_star"] = n_star
        results.append(avg_reward)


def add_bins(subdf):
    subdf['bins'] = pd.cut(results[1].game, range(0, 20_001, 250)).apply(lambda x: x.right)
    return subdf


# +
if not LOAD_RESULTS:
    effect = pd.concat(results)
    effect = effect.groupby("n_star").apply(add_bins).groupby(["n_star", "bins"]).mean()
    effect.to_csv(RESULT_FOLDER + "n_star_effect.csv")
else:
    effect = pd.read_csv(RESULT_FOLDER + 'n_star_effect.csv')
    
effect = effect.drop(["reward_2", "game"], axis=1)
# -

plt.figure(figsize=(20, 6))
vals = pd.pivot(effect.reset_index(), index="n_star", columns="bins", values="reward_1")
g = sns.heatmap(vals)
g.set_xlabel("number of games")
g.set_ylabel("n*")
g.set_title("impact of n* on the average reward of the agent");

# +
plt.figure(figsize=(15, 6))
g = sns.lineplot(data=effect, x="bins", y="reward_1", hue="n_star", palette='magma', legend='full')

g.set_title("effect of $n^*$ on the average reward")
g.set_ylabel("reward")
g.set_xlabel("number of games")
g.set_xlim([0, 23_000])

plt.show()
# -

# ### Question 3

LOAD_RESULTS = False

if not LOAD_RESULTS:
    player_1 = QPlayer(epsilon = lambda n: decreasing_epsilon(n, n_star=1))
    player_2 = OptimalPlayer(epsilon=0.5)
    results = play_n_games(player_1, player_2, n_games=20_000, update_players=1, verbose=False, evaluate=player_1)
    # Group games into bins of 250 games
    results['bins'] = pd.cut(results.game, range(0, 20_001, 250)).apply(lambda x: x.right)

    f = plt.figure(figsize=(15, 6))
    sns.lineplot(data=results[["game", 'player_1_opt', "player_1_rand"]].dropna(), x='game', y="player_1_opt")
    sns.lineplot(data=results[["game", 'player_1_opt', "player_1_rand"]].dropna(), x='game', y="player_1_rand")

# +
n_stars = [1, 10_000, 20_000, 40_000]

if not LOAD_RESULTS:    
    res = []
    player_2 = OptimalPlayer(epsilon=0.5)
    for n_star in n_stars:
        player_1 = QPlayer(epsilon = lambda n: decreasing_epsilon(n, n_star=n_star))
        temp_res = play_n_games(player_1, player_2, n_games=20_000, update_players=1, verbose=False, evaluate=player_1)
        temp_res["n_star"] = n_star
        res.append(temp_res)

    n_star_eval = pd.concat(res)
    n_star_eval = n_star_eval.dropna().drop(["reward_1", "reward_2"], axis=1)

    n_star_eval.to_csv(RESULT_FOLDER + "n_star_eval_rand_opt.csv")
else:
    n_star_eval = pd.read_csv(RESULT_FOLDER + "n_star_eval_rand_opt.csv")

# +
f, a = plt.subplots(figsize=(16, 5))
for n_star in n_stars :
    g = sns.lineplot(data=n_star_eval[n_star_eval.n_star == n_star], x="game", y="player_1_opt", label=f"n_star={n_star}")
    g.set_ylabel('$M_{opt}$', fontsize=16)
plt.show()

f, a = plt.subplots(figsize=(16, 5))
for n_star in n_stars :
    g = sns.lineplot(data=n_star_eval[n_star_eval.n_star == n_star], x="game", y="player_1_rand", label=f"n_star={n_star}")
    g.set_ylabel('$M_{rand}$', fontsize=16)
plt.show()
# -

# ### Question 4
#
# We use different values of $\epsilon_{opt}$ :

eps_opts = (np.arange(10) / 10)[::2]
eps_opts

# +
## optimal n*
opt_n_star = 4_000

## store the differnet results
eps_res = []
## running multiple games for different values of eps_opt
for eps in eps_opts :
    print("Optimal player epsilon:", eps)
    player_1 = QPlayer(epsilon = lambda n: decreasing_epsilon(n, n_star=opt_n_star))
    player_2 = OptimalPlayer(epsilon=eps)
    temp_res = play_n_games(player_1, player_2, n_games=20_000, update_players=1, evaluate=player_1)
    temp_res["eps"] = eps
    eps_res.append(temp_res)
# -




# ### Question 5
#
# The highest $M_{opt}$ achieved in 20 000 games is 0 and $M_{rand}$ is 0.9
#
#
# ### Question 6
#
# Given that there is exists an optimal strategy, there also exists optimal Q-values which are unique up to there ordering (for a given state, the action order is the same for all optimal Q-values).
#
# ### Question 7

LOAD_RESULTS = False

# +
## build a Qplayer that you are going to train against itself
player = QPlayer(epsilon=0.4)

## make it learn by playing against itself
autotrain_result = play_n_games(player, player, n_games=20_000, update_players="both", evaluate=player)
# -

sns.lineplot(data=autotrain_result.dropna(), y="player_1_rand", x="game", label="M_rand")
sns.lineplot(data=autotrain_result.dropna(), y="player_1_opt", x="game", label="M_opt", color="orange")
plt.legend()
plt.show()

# +
eps_opts = [0, 0.2, 0.4, 0.5]

if not LOAD_RESULTS:
    ## store the different results
    eps_res = []
    ## running multiple games for different values of eps_opt
    for eps in eps_opts :
        player = QPlayer(epsilon=eps)
        temp_res = play_n_games(player, player, n_games=20_000, update_players="both", evaluate=player)
        temp_res["eps"] = eps
        eps_res.append(temp_res)
    autotrain_eps = pd.concat(eps_res).dropna()
    autotrain_eps.to_csv(RESULT_FOLDER + "question7.csv")
else:
    autotrain_eps = pd.read_csv(RESULT_FOLDER + "question7.csv", index_col=0)
# -

autotrain_eps.player_1_rand.max(), autotrain_eps.player_1_opt.max()

f, a = plt.subplots(2, 2, figsize=(15, 10))
for eps, ax in zip(eps_opts, a.flatten()):
    sns.lineplot(data=autotrain_eps[autotrain_eps.eps == eps], y="player_1_rand", x="game", label="M_rand", ax=ax)
    sns.lineplot(data=autotrain_eps[autotrain_eps.eps == eps], y="player_1_opt", x="game", label="M_opt", ax=ax, color="orange")
    ax.set_title(f"epsilon = {eps}")
plt.legend()
plt.show()


# ### Question 8

LOAD_RESULT = False

if not LOAD_RESULT:
    results = []

    for n_star in ns_star :
        print(f">> n* = {n_star}")
        player = QPlayer(epsilon=lambda n: decreasing_epsilon(n, n_star = n_star))
        result = play_n_games(player, player, n_games=20_000, update_players="both", verbose=False, evaluate=player)
        result["n_star"] = n_star
        results.append(result)

    effect = pd.concat(results).dropna()
    effect.to_csv(RESULT_FOLDER + "question8.csv")
else:
    effect = pd.read_csv(RESULT_FOLDER + "question8.csv")

f, a = plt.subplots(2, 2, figsize=(15, 10))
for n_star, ax in zip(ns_star, a.flatten()):
    sns.lineplot(data=effect[effect.n_star == n_star], x="game", y="player_1_rand", ax=ax, label="rand")
    sns.lineplot(data=effect[effect.n_star == n_star], x="game", y="player_1_opt", ax=ax, label="opt", color="orange")
    ax.set_ylabel("metric value")
    ax.set_title(f"n* = {n_star}")
plt.legend()
plt.show()

# ### Question 9
#
# What are the highest values of Mopt and Mrand that you could achieve after playing 20’000
# games?
#
# M_opt: -1
#
# M_rand:0.77
#
#
# ### Question 10
#
# For three board arrangements (i.e. states s), visualize Q-values of available actions (e.g.
# using heat maps). Does the result make sense? Did the agent learn the game well?
# Expected answer: A figure with 3 subplots of 3 different states with Q-values shown at available actions
# (caption length < 200 words).

for state, q_val in player.q_values.items():
    if sum(abs(np.array(state))) != 8:
        if any(q_val > 0.025):
            winning_state = state
        if any(q_val < -0.025):
            losing_state = state
        if sum(q_val != 0) >= 2 and all(abs(q_val) < 0.01):
            unsure_state = state

player.set_player('X')
for state in [winning_state, losing_state, unsure_state]:
    sns.heatmap(player.q_values['X'][state].reshape((3,3)), cmap='Blues', annot=np.array(state).reshape((3,3)))
    plt.show()

# # DQN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque, namedtuple

# +
Observation = namedtuple('Observation', 'state action next_state reward')

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.buffer = deque([], maxlen=buffer_size)
        self.batch_size = batch_size  # Size of batch taken from replay buffer

    def length(self):
        return len(self.buffer)

    def can_update(self):
        return self.length() >= self.batch_size # Check if we have a complete batch of observations

    def append(self, *args):
        self.buffer.append(Observation(*args))

    def random_sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        #Unzip observation values into separate lists and cast each list as a numpy array
        return [np.array(val) for val in zip(*sample)]


# -

class DQNPlayer:

    def __init__(self, epsilon=0.2, alpha=0.05, gamma=0.99, batch_size=64, buffer_size=10_000, 
                update_target_frequency=500, lr=0.0005, debug=False, seed=42):
        self.training = True
        self.debug = debug
        self.alpha = alpha
        self.gamma = gamma
        self.seed = seed

        # By default epsilon is a function of the game number n
        # For a constant epsilon we thus convert it to a constant function of n
        self.epsilon = epsilon
        if type(epsilon) in [int, float]:
            self.epsilon = lambda n: epsilon

        # # We store the last state and action played for each player for a future update
        self.last_action = {'X': None, 'O': None}
        self.last_state =  {'X': None, 'O': None}
        
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.update_target_frequency = update_target_frequency # How often to update the target network
        self.update_counter = 0

        self.optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.loss_function = keras.losses.Huber()
        self.model = DQNPlayer._create_model() # Predicts at each step
        self.target_model = DQNPlayer._create_model() # Target model is updated every `update_target_frequency` steps
        self._update_target() #set the target weights equal to the model weights

    @staticmethod
    def _create_model():
        inputs = layers.Input(shape=(3, 3, 2,))
        layer0 = layers.Flatten()(inputs) # Flatten input tensors
        layer1 = layers.Dense(128, activation="relu")(layer0) # 2 hidden layers with ReLu activation
        layer2 = layers.Dense(128, activation="relu")(layer1)
        action = layers.Dense(9, activation="linear")(layer2)
        return keras.Model(inputs=inputs, outputs=action)

    def train(self):
        """Set training mode, in which random actions are chosen with probability epsilon"""
        self.training = True
    
    def eval(self):
        """Set evaluation mode, in which the best action is systematically chosen"""
        self.training = False

    def toggle_debug(self):
        """Toggle debug mode, which prints Q-values after each update"""
        self.debug = not self.debug
        
    @staticmethod    
    def _invalid_action_indices(grid):
        """Returns indices of unallowed actions given the state of the game"""
        grid = grid.copy()
        grid = tf.reduce_sum(grid, axis=-1).numpy()
        return np.where(grid.flatten() != 0)[0]
    
    @staticmethod    
    def _valid_action_indices(grid, tensor=False):
        """Returns indices of allowed actions given the state of the game"""
        grid = grid.copy()
        if tensor:
            grid = tf.reduce_sum(grid, axis=-1).numpy()
        return np.where(grid.flatten() == 0)[0]
    

    @staticmethod
    def _state2input(grid, player):
        """Returns a 3x3x2 tensor from the 2d grid"""
        player_val = 1 if player == 'X' else -1 # Find which value the player is
        opponent_val = -1 * player_val

        t = np.zeros((3,3,2))
        t[:,:,0] = grid == player_val # Set to 1 the position taken by the player
        t[:,:,1] = grid == opponent_val # Set to 1 the position taken by the opponent
        return t.astype(int)

    def _random_move(self, grid):
        """Returns a random available action"""
        return int(np.random.choice(DQNPlayer._valid_action_indices(grid)))
        #return random.randrange(9)

    def _predict_move(self, grid, player):
        """
        Given the 9-action array associated to the current grid, 
        pick the move with highest Q-value
        """
        # Convert the grid to an input tensor of shape (1, 3, 3, 2)
        state_tensor = DQNPlayer._state2input(grid, player)[None,:,:,:] 
        # Predict output
        action_probs = self.model(state_tensor, training=False)
        # Choose the action with the highest output
        action_chosen = tf.argmax(action_probs[0]).numpy() 
        return int(action_chosen)
    
    def act(self, grid, player='X', n=0):
        """Choose the next move according to the state of the game"""
        action = None
        if self.training:
            if random.random() < self.epsilon(n): # explore new action with probability epsilon
                # random move
                action = self._random_move(grid)
            else :
                action = self._predict_move(grid, player)
        else: # if not in training, choose greedy action
            action = self._predict_move(grid, player)
        
        # Save the current state and action for future update
        self.last_action[player] = action
        self.last_state[player] = grid

        return action
    
    def _update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def update(self, current_state, reward, player):
        """
        Update the Q-value associated with the last state and action performed
        according to the Q-learning update rule.
        
        current_state: grd after performing the last action
        reward: total reward obtained by this player after its last action and the following opponent's action
        """
        
        self.replay_buffer.append(DQNPlayer._state2input(self.last_state[player], player),
                        self.last_action[player], 
                        DQNPlayer._state2input(current_state, player), 
                        reward)

        if not self.replay_buffer.can_update(): # Return if we don't have enough observations yet
            return 0#, 0, 0

        states, actions, next_states, rewards = self.replay_buffer.random_sample()
        
        # Get the updated Q-values for the sampled future states
        future_rewards = self.target_model.predict(next_states) # batch size x 9
        # Create a mask of states that are not final
        non_final_mask = tf.convert_to_tensor((np.array(rewards) == 0).astype(float))
        # Set future rewards to 0 if it is already in a final state
        future_rewards *= non_final_mask[:,None]
        
        #set illegal actions to -1 in legal_rewards
        legal_rewards = np.clip(future_rewards.numpy(), -1, 1) # clip predicted future rewards in [-1, 1]
        for i, next_state in enumerate(next_states):
            #legal_rewards[i] = future_rewards[i]
            legal_rewards[i, DQNPlayer._invalid_action_indices(next_state)] = -1 
            #legal_action_mask = np.ones(9)
            #legal_action_mask[DQNPlayer._invalid_action_indices(next_state)] = -1
            #legal_rewards[i] = future_rewards[i] * legal_action_mask # set rewards of illegal actions to -1 
        
        # Q value = reward + discount factor * expected future reward
        target_q_values = rewards + self.gamma * tf.reduce_max(legal_rewards,axis=1) #tf.clip_by_value( ,-1,1)
        # Create a mask so we only calculate loss on the updated Q-values
        chosen_action_mask = tf.one_hot(actions, 9) # batch size x 9 
        
        with tf.GradientTape() as tape:
            # Get the 9 predicted Q-values for each state
            q_values = self.model(states) # batch size x 9

            # Only keep the Q-value for the action chosen
            current_q_values = tf.reduce_sum(q_values * chosen_action_mask, axis=1)
            # Calculate loss between target Q-values and predicted Q-values
            loss = self.loss_function(target_q_values, current_q_values)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
       
        # Update the target network every `update_target_frequency` games
        self.update_counter += 1
        if self.update_counter == self.update_target_frequency:
            self._update_target()
            self.update_counter = 0
        return loss.numpy()#, q_values.numpy().min(), q_values.numpy().max()


# +
def play_n_games(player_1, player_2, n_games=20_000, update_players=None, verbose=False, evaluate=None, seed=42):
    """
    Play a specified number of tic tac toe games between two players.
    
    player_1: first player. Needs to implement methods `act` and `set_player`
    player_2: second player. Needs to implement methods `act` and `set_player`
    n_games: number of games played between the two players
    update_players: determines which players need to be updated after each move. Can take value 1, 2 or 'both'.
    """
    np.random.seed(seed)
    env = TictactoeEnv()
    turns = np.array(['O', 'X'])
    results, evalutions = [], []
    reward_1 = reward_2 = 0
    loss_1 = loss_2 = 0
    #rolling_win_average = n_illegal_move = loss = min_target= max_target= counter = 0
    for game_number in tqdm(range(n_games)):
        env.reset()
        grid, _, __ = env.observe()
        turns = turns[[1,0]] # Swap X and O every game

        for j in range(9):
            is_player_1_move = env.current_player == turns[0]
            if is_player_1_move:
                move = player_1.act(grid, player=turns[0], n=game_number)
            else:
                move = player_2.act(grid, player=turns[1], n=game_number)
            

            if env.check_valid(move):
                grid, end, winner = env.step(move, print_grid=verbose)
                reward_1 = env.reward(turns[0])
                reward_2 = -1 * reward_1
            else:
                #n_illegal_move += 1
            # If a move is not valid then the current player lost the game
                end = True
                winner = turns[1] if is_player_1_move else turns[0]
                reward_1 = -1 if is_player_1_move else 1
                reward_2 = -1 * reward_1

            if verbose :
                print(f"Reward 1: {reward_1}")
                print(f"Reward 2: {reward_2}")
                
            # Check if we need to update any player's parameters
            if (end or j >= 2) and update_players is not None:
                
                # Update player 1's Q-value if player 2 just played or if it's the end
                if (end or not is_player_1_move) and update_players in [1, 'both']:
                    if verbose :
                        print(f"Updating {player_1} with reward {reward_1}")
                    loss_1 = player_1.update(grid, reward_1, turns[0])#,  upd_min_target, upd_max_target
                    #loss += upd_loss
                    #min_target += upd_min_target
                    #max_target += upd_max_target
                    #counter+= 1
                    
                # Update player 2's Q-value if player 1 just played or if it's the end
                if (end or is_player_1_move) and update_players in [2, 'both']:
                    if verbose :
                        print(f"Updating {player_2} with reward {reward_2}")
                    loss_2 = player_2.update(grid, reward_2, turns[1])

            if end:
                if verbose:
                    print('-------------------------------------------')
                    print('Game end, winner is player ' + str(winner))
                    print('Player 1 = ' +  turns[0])
                    print('Player 2 = ' +  turns[1])
                
#                rolling_win_average += reward_1
#                if (game_number + 1) % 250 == 0:
#                    print(f"win rate: {rolling_win_average / 250:.2f}, {n_illegal_move} illegal moves, \
#number of updates: {counter}, loss_1: {loss_1/counter:.4f}, min q-value: {min_target/counter:.2f}, max q-value: {max_target/counter:.2f}", end='\r')
#                    rolling_win_average = n_illegal_move = loss = min_target =max_target = counter = 0

                # Compute M_opt and M_rand on the specified players every 250 games
                if evaluate is not None and (game_number + 1) % 250 == 0:
                    evaluation = [game_number]
                    
                    if evaluate == player_1 or evaluate=="both":
                        player_1.eval()
                        m_opt = measure_score(player_1, "opt", verbose=False)
                        m_rand = measure_score(player_1, "rand", verbose=False)
                        evaluation.append(m_opt)
                        evaluation.append(m_rand)
                        player_1.train()
                    
                    # Don't evalute twice if training against itself
                    if (evaluate == player_2 or evaluate=="both") and player_1 != player_2:
                        player_2.eval()
                        m_opt = measure_score(player_2, "opt", verbose=False)
                        m_rand = measure_score(player_2, "rand", verbose=False)
                        evaluation.append(m_opt)
                        evaluation.append(m_rand)
                        player_2.train()
                        
                    evalutions.append(evaluation)
                    
                results.append([game_number + 1, reward_1, reward_2, turns[0], turns[1], loss_1, loss_2])
                break
    
    game_res = pd.DataFrame(data=results, columns=['game', 'reward_1', 'reward_2', 'player_1', 'player_2', 'loss_1', 'loss_2'])
    perf_columns = ["game"]
    
    if evaluate == player_1 or evaluate == "both":
        perf_columns += ["player_1_opt", "player_1_rand"]
    if (evaluate == player_2 or evaluate == "both") and player_1 != player_2:
        perf_columns += ["player_2_opt", "player_2_rand"]
        
    agents_perf = pd.DataFrame(data=evalutions, columns=perf_columns)

    results = game_res if evaluate is None else game_res.merge(agents_perf, how="outer", on="game")
    return results
# -

# ### Question 11
#
# Plot average reward and average training loss for every 250 games during training. Does the loss decrease? Does the agent learn to play Tic Tac Toe?
# Expected answer: A figure with two subplots (caption length < 50 words). Specify your choice of ε.

LOAD_RESULTS = True

n_games = 20_000
if not LOAD_RESULTS:
    #linear activation and clipping only future_rewards
    player_1 = DQNPlayer()
    player_2 = OptimalPlayer(epsilon=0.5)
    results = play_n_games(player_1, player_2, n_games=n_games, update_players=1, verbose=False)
    results.to_csv(RESULT_FOLDER + 'question11.csv', index=False)
else:
    results = pd.read_csv(RESULT_FOLDER + 'question11.csv')
# Group games into bins of 250 games
results['bins'] = pd.cut(results.game, range(0, n_games + 1, 250)).apply(lambda x: x.right)
sns.lineplot(data=results.groupby("bins").mean(), x="game", y="reward_1")
plt.show()
sns.lineplot(data=results.groupby("bins").mean(), x="game", y="loss_1")
plt.show()

# ### Question 12. 
# Repeat the training but without the replay buffer and with a batch size of 1: At every step, update the network by using only the latest transition. What do you observe?
# Expected answer: A figure with two subplots showing average reward and average training loss during training (caption length < 50 words).

LOAD_RESULTS = True

if not LOAD_RESULTS:
    player_1 = DQNPlayer(buffer_size=1, batch_size=1)
    player_2 = OptimalPlayer(epsilon=0.5)
    results = play_n_games(player_1, player_2, n_games=n_games, update_players=1, verbose=False)
    results.to_csv(RESULT_FOLDER + 'question12.csv', index=False)
else:
    results = pd.read_csv(RESULT_FOLDER + 'question12.csv')
# Group games into bins of 250 games
results['bins'] = pd.cut(results.game, range(0, n_games + 1, 250)).apply(lambda x: x.right)
sns.lineplot(data=results.groupby("bins").mean(), x="game", y="reward_1")
plt.show()
sns.lineplot(data=results.groupby("bins").mean(), x="game", y="loss_1")

# ### Question 13
#
# Instead of fixing ε, use ε(n) in Equation 1. For different values of n∗, run your DQN against Opt(0.5) for 20’000 games – switch the 1st player after every game. Choose several values of n∗ from a reasonably wide interval between 1 to 40’000 – particularly, include $n^∗$ = 1.
#
# After every 250 games during training, compute the ‘test’ Mopt and Mrand for your agents. Plot Mopt and Mrand over time. Does decreasing ε help training compared to having a fixed ε? What is the effect of n∗?

LOAD_RESULTS = True

n_stars = np.arange(0,42_000,8_000)
n_stars[0] = 1
n_stars

if not LOAD_RESULTS:    
    res = []
    player_2 = OptimalPlayer(epsilon=0.5)
    for n_star in n_stars:
        player_1 = DQNPlayer(epsilon = lambda n: decreasing_epsilon(n, n_star=n_star))
        temp_res = play_n_games(player_1, player_2, n_games=20_000, update_players=1, verbose=False, evaluate=player_1)
        temp_res["n_star"] = n_star
        res.append(temp_res)

    n_star_eval = pd.concat(res)
    n_star_eval = n_star_eval.dropna().drop(["reward_1", "reward_2"], axis=1)

    n_star_eval.to_csv(RESULT_FOLDER + "question13.csv")
else:
    n_star_eval = pd.read_csv(RESULT_FOLDER + "question13.csv")

# +
f, a = plt.subplots(figsize=(16, 5))
for n_star in n_stars :
    g = sns.lineplot(data=n_star_eval[n_star_eval.n_star == n_star], x="game", y="player_1_opt", label=f"n_star={n_star}")
    g.set_ylabel('$M_{opt}$', fontsize=16)
plt.show()

f, a = plt.subplots(figsize=(16, 5))
for n_star in n_stars :
    g = sns.lineplot(data=n_star_eval[n_star_eval.n_star == n_star], x="game", y="player_1_rand", label=f"n_star={n_star}")
    g.set_ylabel('$M_{rand}$', fontsize=16)
plt.show()
# -

# optimal n* may be 1 as it seems to converge towards 0 at the end or 32000 which seems to converge faster towards 0 even though the last value is at -1

# ### Question 14
# Choose the best value of n∗ that you found. Run DQN against $Opt(ε_{opt})$ for different values of εopt for 20’000 games – switch the 1st player after every game. Choose several values of εopt from a reasonably wide interval between 0 to 1 – particularly, include εopt = 0.
#
#
# After every 250 games during training, compute the ‘test’ Mopt and Mrand for your agents 4
# – for each value of εopt. Plot Mopt and Mrand over time. What do you observe? How can you explain it? Expected answer: A figure showing Mopt and Mrand over time for different values of εopt (caption length < 250 words).

LOAD_RESULTS = True

eps_opts = (np.arange(10) / 10)[::2] 
eps_opts

# +
## optimal n*
opt_n_star = 32_000

if not LOAD_RESULTS:
    ## store the differnet results
    eps_res = []
    ## running multiple games for different values of eps_opt
    for eps in eps_opts :
        print("Optimal player epsilon:", eps)
        player_1 = DQNPlayer(epsilon=lambda n: decreasing_epsilon(n, n_star=opt_n_star))
        player_2 = OptimalPlayer(epsilon=eps)
        temp_res = play_n_games(player_1, player_2, n_games=20_000, update_players=1, evaluate=player_1)
        temp_res["eps"] = eps
        eps_res.append(temp_res)

        eps_eval = pd.concat(eps_res)
        eps_eval = eps_eval.dropna(subset=['player_1_opt', 'player_1_rand']).drop(["reward_1", "reward_2"], axis=1)

        eps_eval.to_csv(RESULT_FOLDER + "question14.csv")
else:
    eps_eval = pd.read_csv(RESULT_FOLDER + "32000_question14.csv")
# +
#nstar = 1
f, a = plt.subplots(figsize=(16, 5))
for eps in eps_opts :
    g = sns.lineplot(data=eps_eval[eps_eval.eps == eps], x="game", y="player_1_opt", label=f"$\epsilon=${eps}")
    g.set_ylabel('$M_{opt}$', fontsize=16)
plt.show()

f, a = plt.subplots(figsize=(16, 5))
for eps in eps_opts :
    g = sns.lineplot(data=eps_eval[eps_eval.eps == eps], x="game", y="player_1_rand", label=f"$\epsilon=${eps}")
    g.set_ylabel('$M_{rand}$', fontsize=16)
plt.show()

# +
#nstar = 32 000
f, a = plt.subplots(figsize=(16, 5))
for eps in eps_opts :
    g = sns.lineplot(data=eps_eval[eps_eval.eps == eps], x="game", y="player_1_opt", label=f"$\epsilon=${eps}")
    g.set_ylabel('$M_{opt}$', fontsize=16)
plt.show()

f, a = plt.subplots(figsize=(16, 5))
for eps in eps_opts :
    g = sns.lineplot(data=eps_eval[eps_eval.eps == eps], x="game", y="player_1_rand", label=f"$\epsilon=${eps}")
    g.set_ylabel('$M_{rand}$', fontsize=16)
plt.show()
# -

# ### Question 15
# What are the highest values of Mopt and Mrand that you could achieve after playing 20’000 games?
#



# ### Question 16
# For different values of ε ∈ [0, 1), run a DQN agent against itself for 20’000 games – i.e. both players use the same neural network and share the same replay buffer.
#
# After every 250 games during training, compute the ‘test’ Mopt and Mrand for different values of ε ∈ [0, 1). Plot Mopt and Mrand over time. Does the agent learn to play Tic Tac Toe? What is the effect of ε?
# Expected answer: A figure showing Mopt and Mrand over time for different values of ε ∈ [0, 1) (caption length < 100 words).

LOAD_RESULTS = True

# +
eps_opts = [0, 0.2, 0.4, 0.5]

if not LOAD_RESULTS:
    ## store the different results
    eps_res = []
    ## running multiple games for different values of eps_opt
    for eps in eps_opts :
        player = DQNPlayer(epsilon=eps)
        temp_res = play_n_games(player, player, n_games=20_000, update_players="both", evaluate=player)
        temp_res["eps"] = eps
        eps_res.append(temp_res)
    autotrain_eps = pd.concat(eps_res).dropna()
    autotrain_eps.to_csv(RESULT_FOLDER + "question16.csv")
else:
    autotrain_eps = pd.read_csv(RESULT_FOLDER + "question16.csv", index_col=0)
# -

autotrain_eps.player_1_rand.max(), autotrain_eps.player_1_opt.max()

f, a = plt.subplots(2, 2, figsize=(15, 10))
for eps, ax in zip(eps_opts, a.flatten()):
    sns.lineplot(data=autotrain_eps[autotrain_eps.eps == eps], y="player_1_rand", x="game", label="M_rand", ax=ax)
    sns.lineplot(data=autotrain_eps[autotrain_eps.eps == eps], y="player_1_opt", x="game", label="M_opt", ax=ax, color="orange")
    ax.set_title(f"epsilon = {eps}")
plt.legend()
plt.show()

# Question 17. 
#
# Instead of fixing ε, use ε(n) in Equation 1 with different values of n∗.
#
# After every 250 games during training, compute the ‘test’ Mopt and Mrand for your agents. Plot Mopt and Mrand over time. Does decreasing ε help training compared to having a fixed ε? What is the effect of n∗?
# Expected answer: A figure showing Mopt and Mrand over time for different values of speeds of n∗ (caption length < 100 words).


LOAD_RESULT = True

if not LOAD_RESULT:
    results = []
    for n_star in n_stars :
        print(f">> n* = {n_star}")
        player = DQNPlayer(epsilon=lambda n: decreasing_epsilon(n, n_star = n_star))
        result = play_n_games(player, player, n_games=20_000, update_players="both", verbose=False, evaluate=player)
        result["n_star"] = n_star
        results.append(result)
    effect = pd.concat(results).dropna()
    effect.to_csv(RESULT_FOLDER + "question17.csv")
else:
    effect = pd.read_csv(RESULT_FOLDER + "question17.csv")

f, a = plt.subplots(3, 2, figsize=(15, 15), sharey=True)
for n_star, ax in zip(n_stars, a.flatten()):
    sns.lineplot(data=effect[effect.n_star == n_star], x="game", y="player_1_rand", ax=ax, label="rand")
    sns.lineplot(data=effect[effect.n_star == n_star], x="game", y="player_1_opt", ax=ax, label="opt", color="orange")
    ax.set_ylabel("metric value")
    ax.set_title(f"n* = {n_star}")
plt.legend()
plt.show()


