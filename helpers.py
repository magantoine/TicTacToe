import numpy as np
import pandas as pd
from tqdm import tqdm
from tic_env import TictactoeEnv, OptimalPlayer
from collections import defaultdict
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



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
            

            grid, end, winner = env.step(move)

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





class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.action_history = [] 
        self.state_history = [] 
        self.next_state_history = []
        self.rewards_history = []
        self.done_history = []
        self.batch_size = batch_size  # Size of batch taken from replay buffer
        self.buffer_size = buffer_size # Maximum replay length

    def length(self):
        return len(self.done_history)

    def can_update(self):
        return self.length() >= self.batch_size # Check if we have a complete batch of observations

    def append(self, last_action, last_state, current_state, reward):
        # Limit the state and reward history
        if self.length() > self.buffer_size:
            del self.rewards_history[:1]
            del self.state_history[:1]
            del self.next_state_history[:1]
            del self.action_history[:1]
            del self.done_history[:1]

        self.action_history.append(last_action)
        self.state_history.append(last_state)
        self.next_state_history.append(current_state)
        self.done_history.append(reward != 0)
        self.rewards_history.append(reward)

    
    def random_sample(self):
        # Get random indices of samples for replay buffers
        indices = np.random.choice(range(self.length()), size=self.batch_size)

        # sample buffers
        state_sample = np.array([self.state_history[i] for i in indices])
        state_next_sample = np.array([self.next_state_history[i] for i in indices])
        rewards_sample = np.array([self.rewards_history[i] for i in indices])
        action_sample = np.array([self.action_history[i] for i in indices])
        done_sample = tf.convert_to_tensor([float(self.done_history[i]) for i in indices])[:,None]
        return state_sample, state_next_sample, rewards_sample, action_sample, done_sample




class DQNPlayer:

    def __init__(self, epsilon=0.2, alpha=0.05, gamma=0.99, batch_size=64, buffer_size=10_000, 
                update_target_frequency=500, learning_rate=0.0005, debug=False, seed=42, ):
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
        self.update_count = 0

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        self.loss_function = keras.losses.Huber()
        self.model = DQNPlayer._create_model() # Predicts at each step
        self.target_model = DQNPlayer._create_model() # Target model is updated every `update_target_frequency` steps

    @staticmethod
    def _create_model():
        inputs = layers.Input(shape=(3, 3, 2,))
        layer0 = layers.Flatten()(inputs) # Flatten input tensores
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
    def _valid_action_indices(grid):
        """Returns indices of allowed actions given the state of the game"""
        return np.where(grid.flatten() == 0)[0]

    @staticmethod
    def _state2tensor(grid, player):
        """Returns a 3x3x2 tensor from the 2d grid"""
        player_val = 1 if player == 'X' else -1
        opponent_val = -1 * player_val

        t = np.zeros((3,3,2))
        t[:,:,0] = (grid == player_val).astype(int)
        t[:,:,1] = (grid == opponent_val).astype(int)

        return tf.convert_to_tensor(t)

    def _random_move(self, grid):
        """Returns a random available action"""
        return int(np.random.choice(DQNPlayer._valid_action_indices(grid)))
        
    def _predict_move(self, grid, player):
        """
        Given the 9-action array associated to the current grid, 
        pick the move with highest Q-value
        """
        state_tensor = DQNPlayer._state2tensor(grid, player)[None,:,:,:] # Convert the grid to the input tensor of shape (1, 3, 3, 2)
        action_probs = self.model(state_tensor) # Predict output, TODO: if we add dropout we can tweak the argument `training=False`
        action_chosen = tf.argmax(action_probs[0]).numpy() # Choose the action with the highest output
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

    def update(self, current_state, reward, player):
        """
        Update the Q-value associated with the last state and action performed
        according to the Q-learning update rule.
        
        current_state: grd after performing the last action
        reward: total reward obtained by this player after its last action and the following opponent's action
        """
        self.replay_buffer.append(self.last_action[player], 
                            DQNPlayer._state2tensor(self.last_state[player], player),
                            DQNPlayer._state2tensor(current_state, player), 
                            reward)

        if not self.replay_buffer.can_update(): # Return if we don't have enough observations yet
            return

        state_sample, state_next_sample,\
                    rewards_sample, action_sample, done_sample = self.replay_buffer.random_sample()

        # Get the updated Q-values for the sampled future states
        future_rewards = self.target_model.predict(state_next_sample) # batch size x 9
        future_rewards *= (1 - done_sample) # set future rewards to 0 if it is already in a final state

        # Q value = reward + discount factor * expected future reward
        target_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)
        
        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, 9) # batch size x 9 

        with tf.GradientTape() as tape:
            # Get the 9 predicted Q-values for each state
            q_values = self.model(state_sample) # batch size x 9

            # Only keep the Q-value for the action chosen
            current_q_values = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(target_q_values, current_q_values)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Update the target network every `update_target_frequency` games
        self.update_count += 1
        if self.update_count == self.update_target_frequency:
            self.target_model.set_weights(self.model.get_weights())
            self.update_count = 0