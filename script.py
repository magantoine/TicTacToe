import dill
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tic_env import OptimalPlayer
from helpers import QPlayer, play_n_games, DQNPlayer, RESULT_FOLDER, N_GAMES, play_n_games_dqn

n_stars = np.arange(0,42_000,8_000)
n_stars[0] = 1
eps_opts = (np.arange(10) / 10)[::2] 

def decreasing_epsilon(n, n_star=10_000, eps_min=0.1, eps_max=0.8):
    return max(eps_min, eps_max * (1 - n / n_star))


def add_bins(subdf):
    subdf['bins'] = pd.cut(subdf.game, range(0, 20_001, 250)).apply(lambda x: x.right)
    return subdf


def question12():
    player_1 = DQNPlayer(buffer_size=1, batch_size=1)
    player_2 = OptimalPlayer(epsilon=0.5)
    results = play_n_games_dqn(player_1, player_2, n_games=N_GAMES, update_players=1, verbose=False)
    results.to_csv(RESULT_FOLDER + 'fast_question12.csv', index=False)

def question13():
    res = []
    player_2 = OptimalPlayer(epsilon=0.6)
    for n_star in n_stars:
        player_1 = DQNPlayer(epsilon = lambda n: decreasing_epsilon(n, n_star=n_star))
        temp_res = play_n_games_dqn(player_1, player_2, n_games=20_000, update_players=1, verbose=False, evaluate=player_1)
        temp_res["n_star"] = n_star
        res.append(temp_res)

    n_star_eval = pd.concat(res)
    n_star_eval = n_star_eval.dropna().drop(["reward_1", "reward_2"], axis=1)

    n_star_eval.to_csv(RESULT_FOLDER + "fast_question13.csv")

def question14(opt_n_star):
    eps_res = []
    ## running multiple games for different values of eps_opt
    for eps in eps_opts :
        print("Optimal player epsilon:", eps)
        player_1 = DQNPlayer(epsilon=lambda n: decreasing_epsilon(n, n_star=opt_n_star))
        player_2 = OptimalPlayer(epsilon=eps)
        temp_res = play_n_games_dqn(player_1, player_2, n_games=N_GAMES, update_players=1, evaluate=player_1)
        temp_res["eps"] = eps
        eps_res.append(temp_res)

        eps_eval = pd.concat(eps_res)
        eps_eval = eps_eval.dropna(subset=['player_1_opt', 'player_1_rand']).drop(["reward_1", "reward_2"], axis=1)

        eps_eval.to_csv(RESULT_FOLDER + f"{opt_n_star}_fast_question14.csv")



def question16():
    eps_opts = [0, 0.2, 0.4, 0.5]
    ## store the different results
    eps_res = []
    ## running multiple games for different values of eps_opt
    for eps in eps_opts :
        player = DQNPlayer(epsilon=eps)
        temp_res = play_n_games_dqn(player, player, n_games=N_GAMES, update_players="both", evaluate=player)
        temp_res["eps"] = eps
        eps_res.append(temp_res)
    autotrain_eps = pd.concat(eps_res).dropna()
    autotrain_eps.to_csv(RESULT_FOLDER + "fast_question16.csv")

def question17():
    results = []
    for n_star in n_stars :
        print(f">> n* = {n_star}")
        player = DQNPlayer(epsilon=lambda n: decreasing_epsilon(n, n_star = n_star))
        result = play_n_games_dqn(player, player, n_games=N_GAMES, update_players="both", verbose=False, evaluate=player)
        result["n_star"] = n_star
        results.append(result)
    effect = pd.concat(results).dropna()
    effect.to_csv(RESULT_FOLDER + "fast_question17.csv")

def question19():
    player = DQNPlayer(epsilon=lambda n: decreasing_epsilon(n, n_star = 24_000))
    result = play_n_games_dqn(player, player, n_games=N_GAMES, update_players="both", verbose=False)
    with open(join(RESULT_FOLDER,'dqn_player.pkl'), "wb") as f:
        dill.dump(player, f)

def main():
    print(">>> Question 19")
    question19()

if __name__ == '__main__':
    main()