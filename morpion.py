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

# %load_ext autoreload
# %autoreload 2

import dill
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from tic_env import OptimalPlayer
from helpers import QPlayer, play_n_games, DQNPlayer, RESULT_FOLDER, N_GAMES,\
play_n_games_dqn


# $Q(S, A) ← Q(S, A) + α(R + γ max_a Q(S′, a) − Q(S, A))$

def decreasing_epsilon(n, n_star=10_000, eps_min=0.1, eps_max=0.8):
    return max(eps_min, eps_max * (1 - n / n_star))


def add_bins(subdf):
    subdf['bins'] = pd.cut(subdf.game, range(0, 20_001, 250)).apply(lambda x: x.right)
    return subdf


# ## 2.1 Learning from experts
#
# In this section, you will study whether Q-learning can learn to play Tic Tac Toe by playing against
# Opt($\epsilon$opt) for some $\epsilon$opt ∈ [0, 1]. To do so, implement the Q-learning algorithm. To check the algorithm,
# run a Q-learning agent, with a fixed and arbitrary $\epsilon \in [0, 1)$, against Opt(0.5) for 20’000 games – switch
# the 1st player after every game.
#
# ### Question 1:
#
# Plot average reward for every 250 games during training – i.e. after the 50th game, plot the average reward of the first 250 games, after the 100th game, plot the average reward of games 51 to 100, etc. Does the agent learn to play Tic Tac Toe?

# +
## player we will train, default epsilon = 0.2
player_1 = QPlayer()

## opponent, optimal player with 0.5 as epsilon_opt
player_2 = OptimalPlayer(epsilon=0.5)

results = play_n_games(player_1, player_2, n_games=20_000, update_players=1, verbose=False)

# Group games into bins of 250 games
results['bins'] = pd.cut(results.game, range(0, 20_001, 250)).apply(lambda x: x.right)
g = sns.lineplot(data=results.groupby("bins").mean(), x="game", y="reward_1")
g.set_title("Average reward every 250 games")
g.set_ylabel("average reward")

plt.savefig("plots/question1.jpg", dpi=100)
# -

# ### Question 2

LOAD_RESULTS = False

# + pycharm={"name": "#%%\n"}
## player to train with decreasing epsilon
player_1 = QPlayer(epsilon=decreasing_epsilon)
player_2 = OptimalPlayer(epsilon=0.5)
results = play_n_games(player_1, player_2, n_games=20_000, update_players=1, verbose=False)

# +
# Group games into bins of 250 games
results['bins'] = pd.cut(results.game, range(0, 20_001, 250)).apply(lambda x: x.right)
g = sns.lineplot(data=results.groupby("bins").mean(), x="game", y="reward_1")
g.set_ylabel("Average reward")
g.set_title("Average reward every 250 games")

plt.savefig("plots/question2.jpg")
# -

# Let's try several values of $n^*$ to see its effect. We use these values of $n^*$ :

ns_star = np.arange(0,42_000,8_000)
ns_star[0] = 1
ns_star

LOAD_RESULTS = False

if not LOAD_RESULTS:
    results = []

    for n_star in ns_star :
        print(f">> n* = {n_star}")
        player_1 = QPlayer(epsilon = lambda n: decreasing_epsilon(n, n_star=n_star))
        player_2 = OptimalPlayer(epsilon=0.5)
        avg_reward = play_n_games(player_1, player_2, n_games=20_000, update_players=1, verbose=False)
        avg_reward["n_star"] = n_star
        results.append(avg_reward)

# +
if not LOAD_RESULTS:
    effect = pd.concat(results)
    effect = effect.groupby("n_star").apply(add_bins).groupby(["n_star", "bins"]).mean()
    effect.to_csv(RESULT_FOLDER + "n_star_effect.csv")
else:
    effect = pd.read_csv(RESULT_FOLDER + 'n_star_effect.csv')
    
effect = effect.drop(["reward_2", "game"], axis=1)

# +
plt.figure(figsize=(20, 6))
vals = pd.pivot(effect.reset_index(), index="n_star", columns="bins", values="reward_1")
g = sns.heatmap(vals)
g.set_xlabel("number of games")
g.set_ylabel("n*")
g.set_title("impact of n* on the average reward of the agent")

plt.savefig("plots/question2_2.jpg");

# +
plt.figure(figsize=(15, 6))
g = sns.lineplot(data=effect, x="bins", y="reward_1", hue="n_star", palette='magma', legend='full')

g.set_title("effect of $n^*$ on the average reward")
g.set_ylabel("reward")
g.set_xlabel("number of games")
g.set_xlim([0, 23_000])
plt.savefig("plots/question_2_3.jpg")
plt.show()
# -

# ### Question 3

LOAD_RESULTS = True

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
plt.savefig("plots/question3_opt.jpg")

f, a = plt.subplots(figsize=(16, 5))
for n_star in n_stars :
    g = sns.lineplot(data=n_star_eval[n_star_eval.n_star == n_star], x="game", y="player_1_rand", label=f"n_star={n_star}")
    g.set_ylabel('$M_{rand}$', fontsize=16)
plt.show()
plt.savefig("plots/question3_rand.jpg")
# -

# ### Question 4
#
# We use different values of $\epsilon_{opt}$ :

eps_opts = (np.arange(12) / 10)[::2]
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
f, a = plt.subplots(3, 2, figsize=(15, 10))
for eps_unit_res, ax in zip(eps_res, a.flatten()):
    eps_unit_res['bins'] = pd.cut(eps_unit_res.game, range(0, 20_001, 250)).apply(lambda x: x.right)
    sns.lineplot(data=eps_unit_res, x="game", y="player_1_opt", label="opt", ax=ax)
    sns.lineplot(data=eps_unit_res, x="game", y="player_1_rand", label="rand", color="orange", ax=ax)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(f"against Opt({eps_unit_res.eps.iloc[0]})")
f.text(0.5, 0, "Number of games")
f.text(0, 0.5, "Metric", rotation=90)
f.suptitle("Evolution of $M_{rand}$ and $M_{opt}$ when training against Opt($\epsilon$)")
plt.tight_layout()
plt.savefig("plots/question4.jpg", dpi=700);

# ### Question 5
#
# The highest $M_{opt}$ achieved in 20 000 games is 0.2 and $M_{rand}$ is 0.8
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

# +
sns.lineplot(data=autotrain_result.dropna(), y="player_1_rand", x="game", label="M_rand")
g = sns.lineplot(data=autotrain_result.dropna(), y="player_1_opt", x="game", label="M_opt", color="orange")
g.set_ylabel("metric")
g.set_title("performance of self learning")
plt.legend()

plt.savefig("plots/question7.jpg", dpi=700)
plt.show()

# +
eps_opts = [0, 0.2, 0.4, 0.6, 0.8, 1]

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

# +
f, a = plt.subplots(2, 2, figsize=(15, 10))
for eps, ax in zip(eps_opts, a.flatten()):
    sns.lineplot(data=autotrain_eps[autotrain_eps.eps == eps], y="player_1_rand", x="game", label="M_rand", ax=ax)
    sns.lineplot(data=autotrain_eps[autotrain_eps.eps == eps], y="player_1_opt", x="game", label="M_opt", ax=ax, color="orange")
    ax.set_title(f"$\epsilon$ = {eps}")
    ax.set_xlabel("")
    ax.set_ylabel("")
plt.legend()

f.text(0.5, 0, "Number of games")
f.text(0, 0.5, "Metric", rotation=90)
f.suptitle("Evolution of $M_{rand}$ and $M_{opt}$ when training against Opt($\epsilon$)")

plt.savefig("plots/question7_2.jpg", dpi=700)
plt.show()
# -


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


# ### Question 11
#
# Plot average reward and average training loss for every 250 games during training. Does the loss decrease? Does the agent learn to play Tic Tac Toe?
# Expected answer: A figure with two subplots (caption length < 50 words). Specify your choice of ε.

LOAD_RESULTS = True

fontsize= 16

PLOT_FOLDER = 'plots'

# +
if not LOAD_RESULTS:
    #linear activation and clipping only future_rewards
    player_1 = DQNPlayer()
    player_2 = OptimalPlayer(epsilon=0.5)
    results = play_n_games_dqn(player_1, player_2, n_games=N_GAMES, update_players=1, verbose=False)
    results.to_csv(RESULT_FOLDER + 'question11.csv', index=False)
else:
    results = pd.read_csv(RESULT_FOLDER + 'question11.csv')
# Group games into bins of 250 games
results['bins'] = pd.cut(results.game, range(0, N_GAMES + 1, 250)).apply(lambda x: x.right)

fig, axes = plt.subplots(1,2, figsize=(15,5))
sns.lineplot(data=results.groupby("bins").mean(), x="game", y="reward_1", ax=axes[0])
axes[0].set_ylabel("reward", fontsize=fontsize)
sns.lineplot(data=results.groupby("bins").mean(), x="game", y="loss_1", ax=axes[1])
axes[1].set_ylabel("loss", fontsize=fontsize)
plt.suptitle('Average reward and loss through the training games', fontsize=fontsize)
plt.savefig(join(PLOT_FOLDER, "question11.png"), dpi=200)
plt.show()
# -

# ### Question 12. 
# Repeat the training but without the replay buffer and with a batch size of 1: At every step, update the network by using only the latest transition. What do you observe?
# Expected answer: A figure with two subplots showing average reward and average training loss during training (caption length < 50 words).

LOAD_RESULTS = True

# +
if not LOAD_RESULTS:
    player_1 = DQNPlayer(buffer_size=1, batch_size=1)
    player_2 = OptimalPlayer(epsilon=0.5)
    results = play_n_games_dqn(player_1, player_2, n_games=N_GAMES, update_players=1, verbose=False)
    results.to_csv(RESULT_FOLDER + 'fast_question12.csv', index=False)
else:
    results = pd.read_csv(RESULT_FOLDER + 'fast_question12.csv')
# Group games into bins of 250 games
results['bins'] = pd.cut(results.game, range(0, N_GAMES + 1, 250)).apply(lambda x: x.right)

fig, axes = plt.subplots(1,2, figsize=(15,5))
sns.lineplot(data=results.groupby("bins").mean(), x="game", y="reward_1", ax=axes[0])
axes[0].set_ylabel("reward", fontsize=fontsize)
sns.lineplot(data=results.groupby("bins").mean(), x="game", y="loss_1", ax=axes[1])
axes[1].set_ylabel("loss", fontsize=fontsize)
plt.suptitle('Average reward and loss without replay buffer', fontsize=fontsize)
plt.savefig(join(PLOT_FOLDER, "question12.png"), dpi=200)
plt.show()
# -

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
        temp_res = play_n_games_dqn(player_1, player_2, n_games=N_GAMES, update_players=1, verbose=False, evaluate=player_1)
        temp_res["n_star"] = n_star
        res.append(temp_res)

    n_star_eval = pd.concat(res)
    n_star_eval = n_star_eval.dropna().drop(["reward_1", "reward_2"], axis=1)

    n_star_eval.to_csv(RESULT_FOLDER + "question13.csv")
else:
    n_star_eval = pd.read_csv(RESULT_FOLDER + "fast_question13.csv")


def plot_comparison(df, title, filename, hue):
    palette='Spectral'
    f, axes = plt.subplots(2,1,figsize=(16, 10))
    sns.lineplot(data=df, x="game", y="player_1_opt", hue=hue, ax=axes[0],palette=palette)
    axes[0].set_ylabel('$M_{opt}$', fontsize=fontsize)

    sns.lineplot(data=df, x="game", y="player_1_rand", hue=hue, ax=axes[1], palette=palette)
    axes[1].set_ylabel('$M_{rand}$', fontsize=fontsize)

    plt.suptitle(title, fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(join(PLOT_FOLDER, filename), dpi=200)
    plt.show()


plot_comparison(n_star_eval, r"$M_{opt}$ and $M_{rand}$ for different $n^*$",
                'question13.png', 'n_star')

plot_comparison(n_star_eval.query('n_star == 8000'), 
                r"$M_{opt}$ and $M_{rand}$ for $n^* = 8000$",
                '8000_question13.png', 'n_star')

# optimal n* = 8000 is the only value that converges toward 0 without fluctuations to -1

## optimal n*
opt_n_star = 8000

# ### Question 14
# Choose the best value of n∗ that you found. Run DQN against $Opt(ε_{opt})$ for different values of εopt for 20’000 games – switch the 1st player after every game. Choose several values of εopt from a reasonably wide interval between 0 to 1 – particularly, include εopt = 0.
#
#
# After every 250 games during training, compute the ‘test’ Mopt and Mrand for your agents 4
# – for each value of εopt. Plot Mopt and Mrand over time. What do you observe? How can you explain it? Expected answer: A figure showing Mopt and Mrand over time for different values of εopt (caption length < 250 words).

LOAD_RESULTS = True

eps_opts = (np.arange(10) / 10)[::2] 
eps_opts

if not LOAD_RESULTS:
    ## store the differnet results
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

        eps_eval.to_csv(RESULT_FOLDER + "question14.csv")
else:
    eps_eval = pd.read_csv(RESULT_FOLDER + "8000_fast_question14.csv")
plot_comparison(eps_eval, r"$M_{opt}$ and $M_{rand}$ with $n^* = 8000$ against $Opt(\epsilon) $",
                'question14.png', 'eps')

# ### Question 15
# What are the highest values of Mopt and Mrand that you could achieve after playing 20’000 games?
#

print('Max Mrand:', max(n_star_eval.player_1_rand.max(), eps_eval.player_1_rand.max()))

print('Max Mopt:', max(n_star_eval.player_1_opt.max(), eps_eval.player_1_opt.max()))

# ### Question 16
# For different values of ε ∈ [0, 1), run a DQN agent against itself for 20’000 games – i.e. both players use the same neural network and share the same replay buffer.
#
# After every 250 games during training, compute the ‘test’ Mopt and Mrand for different values of ε ∈ [0, 1). Plot Mopt and Mrand over time. Does the agent learn to play Tic Tac Toe? What is the effect of ε?
# Expected answer: A figure showing Mopt and Mrand over time for different values of ε ∈ [0, 1) (caption length < 100 words).

LOAD_RESULTS = True

eps_opts = [0, 0.2, 0.4, 0.5]

if not LOAD_RESULTS:
    ## store the different results
    eps_res = []
    ## running multiple games for different values of eps_opt
    for eps in eps_opts :
        player = DQNPlayer(epsilon=eps)
        temp_res = play_n_games_dqn(player, player, n_games=N_GAMES, update_players="both", evaluate=player)
        temp_res["eps"] = eps
        eps_res.append(temp_res)
    autotrain_eps = pd.concat(eps_res).dropna()
    autotrain_eps.to_csv(RESULT_FOLDER + "question16.csv")
else:
    autotrain_eps = pd.read_csv(RESULT_FOLDER + "question16.csv", index_col=0)

# +
f, a = plt.subplots(2, 2, figsize=(15, 10), sharey=True, sharex=True)
for eps, ax in zip(eps_opts, a.flatten()):
    sns.lineplot(data=autotrain_eps[autotrain_eps.eps == eps], y="player_1_rand", x="game", label="$M_{rand}$", ax=ax)
    sns.lineplot(data=autotrain_eps[autotrain_eps.eps == eps], y="player_1_opt", x="game", label="$M_{opt}$", ax=ax, color="orange")
    ax.set_title(f"$\epsilon = ${eps}", fontsize=fontsize)
    ax.set_ylabel("$M_{opt}$ vs $M_{rand}$", fontsize=fontsize)
    
plt.suptitle('The effect of $\epsilon$ during self learning', fontsize=fontsize)
fig.tight_layout()
plt.legend()
plt.savefig(join(PLOT_FOLDER, 'question16.png'), dpi=200)
plt.show()
# -

# ## Question 17. 
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
        result = play_n_games_dqn(player, player, n_games=N_GAMES, update_players="both", verbose=False, evaluate=player)
        result["n_star"] = n_star
        results.append(result)
    autotrain_n_star = pd.concat(results).dropna()
    autotrain_n_star.to_csv(RESULT_FOLDER + "question17.csv")
else:
    autotrain_n_star = pd.read_csv(RESULT_FOLDER + "question17.csv")

# +
f, a = plt.subplots(3, 2, figsize=(15, 17), sharey=True, sharex=True)
for n_star, ax in zip(n_stars, a.flatten()):
    sns.lineplot(data=autotrain_n_star[autotrain_n_star.n_star == n_star], x="game", y="player_1_rand", ax=ax, label="rand")
    sns.lineplot(data=autotrain_n_star[autotrain_n_star.n_star == n_star], x="game", y="player_1_opt", ax=ax, label="opt", color="orange")
    ax.set_ylabel("$M_{opt}$ vs $M_{rand}$", fontsize=fontsize)
    ax.set_title(f"n* = {n_star}", fontsize=fontsize)

plt.suptitle('The effect of $n^*$ during self learning', fontsize=fontsize, )
fig.tight_layout()
plt.legend()
plt.savefig(join(PLOT_FOLDER, 'question17.png'), dpi=200)
plt.show()
# -

# ## Question 18

print('Max Mrand:', max(autotrain_n_star.player_1_rand.max(), autotrain_eps.player_1_rand.max()))

print('Max Mopt:', max(autotrain_n_star.player_1_opt.max(), autotrain_eps.player_1_opt.max()))

# ## Question 19

LOAD_RESULTS = True

player = DQNPlayer(epsilon=lambda n: decreasing_epsilon(n, n_star = 24_000))

if not LOAD_RESULTS:
    player = DQNPlayer(epsilon=lambda n: decreasing_epsilon(n, n_star = 24_000))
    result = play_n_games_dqn(player, player, n_games=N_GAMES, update_players="both", verbose=False)
    with open(join(RESULT_FOLDER,'dqn_player.pkl'), "wb") as f:
        dill.dump(player, f)
else:
    with open(join(RESULT_FOLDER,'dqn_player.pkl'), "rb") as f:
        player = dill.load(f)
        player.eval()

state_1 = ([[1, -1, 0],
            [0, 1, -1],
            [0, 0, 0]], 'X', 'Does the model win?')
state_2 = ([[1, -1, 0],
            [0, 1, 0],
            [0, 0, 0]], 'O', 'Does the model block?')
state_3 = ([[1, -1, 0], 
            [1, -1, 0],
            [0, 0, 0]], 'X', 'Does the model win or block?')

fig, axes = plt.subplots(3, 1, figsize=(10,25))
for i, (grid, player_val, caption) in enumerate([state_1, state_2, state_3]):
    q_values = player.get_q_values(grid, player_val).numpy().reshape((3,3))
    annot = np.array(grid).astype(str)
    annot = np.where(annot == '1', 'X', annot)
    annot = np.where(annot == '-1', 'O', annot)
    annot = np.where(annot == '0', '-', annot)
    sns.heatmap(q_values, cmap='Blues',annot=annot, ax=axes[i], fmt='', 
                annot_kws={"size": 25}, xticklabels=False, yticklabels=False)
    
    axes[i].set_title(caption)

# ## Question 20



# For QDN training we don't take the maximum Mopt value obtained because it also has the lowest Mrand. Instead we take the best compromise between Mopt and Mrand, which is attained for epsilon is fixed at 0.6

eps_eval.player_1_rand.max(), eps_eval.player_1_opt.max(),

eps_eval[eps_eval.eps == 0.6].player_1_rand.max(), eps_eval[eps_eval.eps == 0.6].player_1_opt.max(),

eps_eval[(eps_eval.player_1_opt > 0.8 * eps_eval[eps_eval.eps == 0.6].player_1_opt.max()) & 
         (eps_eval.player_1_rand > 0.8 * eps_eval[eps_eval.eps == 0.6].player_1_rand.max())].head()

# DQN from experts T_Train = 6748

autotrain_eps[(autotrain_eps.player_1_rand > 0.8 * autotrain_eps.player_1_rand.max())
              & (autotrain_eps.player_1_opt > 0.8 * autotrain_eps.player_1_opt.max())]

autotrain_eps.player_1_rand.max(), autotrain_eps.player_1_opt.max(),

# DQN self-learning T_Train = 8749


