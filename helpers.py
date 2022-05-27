import numpy as np
import pandas as pd
from tqdm import tqdm
from tic_env import TictactoeEnv, OptimalPlayer
from collections import defaultdict
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join

RESULT_FOLDER ='saved_results/'
N_GAMES = 20_000
fontsize= 16
PLOT_FOLDER = 'plots'

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
    reward_1 = reward_2 = loss_1 = loss_2 = 0
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
                    loss_1 = player_1.update(grid, reward_1, turns[0])
                    
                # Update player 2's Q-value if player 1 just played or if it's the end
                if (end or is_player_1_move) and (update_players == 2 or update_players == 'both'):
                    if verbose :
                        print(f"Updating {player_2} with reward {reward_2}")
                    loss_2 = player_2.update(grid, reward_2, turns[1])

            if end:
                if verbose:
                    print('-------------------------------------------')
                    print('Game end, winner is player ' + str(winner))
                    print('Player 1 = ' +  turns[0])
                    print('Player 2 = ' +  turns[1])
                
                rolling_win_average += reward_1
                if (game_number + 1) % 250 == 0:
                    #print(rolling_win_average / 250, end='\r')
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
    # Group games into bins of 250 games
    results['bins'] = pd.cut(results.game, range(0, n_games + 1, 250)).apply(lambda x: x.right)
    return results


def measure_score(player_1, opponent_strategy='opt', n_games=500, seed=42, verbose=True):
    """
    Play a specified number of tic tac toe games between two players.
    
    player_1: player evaluated
    opponent_strategy: 'opt' or 'rand'
    n_games: number of games played between the two players
    seed: number used to determine randomness of the successive games
    """
    
    future_seed = random.randint(0, 32767)

    player_2 = OptimalPlayer(epsilon = 0 if opponent_strategy == 'opt' else 1 )
    env = TictactoeEnv()
    turns = np.array(['O', 'X'])
    m = 0
    rng = np.random.RandomState(seed)
    iterator = tqdm(range(n_games)) if verbose else range(n_games)
    
    if verbose:
        n_win = n_lose = n_even = 0
        
    for game_number in iterator:

        new_seed = rng.randint(low=0, high=32767)
        tf.keras.utils.set_random_seed(new_seed) # sets seed for python, numpy and tf

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
                reward_1 = env.reward(turns[0])
                reward_2 = -1 * reward_1
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
    
    tf.keras.utils.set_random_seed(future_seed)
    
    if verbose:
        return m / n_games, n_win, n_lose, n_even

    
    return m / n_games


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

    def get_q_values(self, grid, player):
        # Convert the grid to an input tensor of shape (1, 3, 3, 2)
        state_tensor = DQNPlayer._state2input(grid, player)[None,:,:,:] 
        # Predict output
        action_probs = self.model(state_tensor, training=False)[0]
        return action_probs

    def _predict_move(self, grid, player):
        """
        Given the 9-action array associated to the current grid, 
        pick the move with highest Q-value
        """
        action_probs = self.get_q_values(grid, player)
        # Choose the action with the highest output
        action_chosen = tf.argmax(action_probs).numpy() 
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
        future_rewards = self.target_model(next_states).numpy() # batch size x 9
        # Create a mask of states that are not final
        non_final_mask = rewards == 0
        # Set future rewards to 0 if it is already in a final state
        future_rewards *= non_final_mask[:,None]
        
        #set illegal actions to -1 in legal_rewards
        legal_rewards = np.clip(future_rewards, -1, 1) # clip predicted future rewards in [-1, 1]
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


def plot_comparison(df, title, filename, hue):
    palette='Spectral'
    fig, axes = plt.subplots(2,1,figsize=(16, 10))
    sns.lineplot(data=df, x="game", y="player_1_opt", hue=hue, ax=axes[0],palette=palette)
    axes[0].set_ylabel('$M_{opt}$', fontsize=fontsize)

    sns.lineplot(data=df, x="game", y="player_1_rand", hue=hue, ax=axes[1], palette=palette)
    axes[1].set_ylabel('$M_{rand}$', fontsize=fontsize)

    plt.suptitle(title, fontsize=fontsize)
    fig.tight_layout()
    if filename:
        plt.savefig(join(PLOT_FOLDER, filename), dpi=200)
    plt.show()
