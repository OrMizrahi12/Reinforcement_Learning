from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from sarsa import print_values, print_policy
from sarsa import max_dict



"""
Q-learning
============

Intro
------

* Q-learning is a model-free RL algorithm used for solving the control problem, 
  specifically for learning an optimal action-value function (Q-funation)
* Q-learning is an off-policy algorithm, 
  meaning it doesn't necessarily follow the policy it's currently estimating. 

  
components and principles of Q-learning
----------------------------------------

1. Action-Value Function (Q-function):
  - The core idea of Q-learning is to learn an action-value function,
    denoted as Q(s,a), which represents the expected cumulative reward of taking
    action `a` in state `s` and following the optimal policy thereafter.

2. Q-learning Update Rule:
  - The update rule for Q-learning is given by:
  - Q(s,a) ← Q(s,a) + a * [r + y * max[a'] * Q(s ',a') - Q(s,a)]

  
Conclusion
-----------

Q-learning is a powerful and widely used algorithm for learning 
optimal policies in reinforcement learning problems. 
It focuses on estimating the action-value function and updates its 
estimates based on observed rewards and the maximum estimated 
Q-values of future states.

"""


GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def epsilon_greedy(Q, s, eps=0.1):
  if np.random.random() < eps:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)
  else:
    a_opt = max_dict(Q[s])[0]
    return a_opt


if __name__ == '__main__':
  """
  1. Initialize an enviroment
  ----------------------------
  """
  # grid = standard_grid()
  grid = negative_grid(step_cost=-0.1)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)
  
  """
  2. initialize Q(s,a)
  --------------------

  As the Q-learning, the Q table'll updated.  
  """
  
  Q = {}
  states = grid.all_states()
  for s in states:
    Q[s] = {}
    for a in ALL_POSSIBLE_ACTIONS:
      Q[s][a] = 0

  # let's also keep track of how many times Q[s] has been updated
  update_counts = {}
  
  """
  3. Q-learning implementation
  ----------------------------

  """
  # repeat until convergence
  reward_per_episode = []
  for it in range(10000):
    if it % 2000 == 0:
      print("it:", it)

    # begin a new episode
    s = grid.reset()
    episode_reward = 0
    while not grid.game_over():
      # perform action and get next state + reward
      a = epsilon_greedy(Q, s, eps=0.1)
      r = grid.move(a)
      s2 = grid.current_state()

      # update reward
      episode_reward += r

      # update Q(s,a)
      maxQ = max_dict(Q[s2])[1] # get the max value of state given action. 
      Q[s][a] = Q[s][a] + ALPHA*(r + GAMMA*maxQ - Q[s][a]) # update Q

      # we would like to know how often Q(s) has been updated too
      update_counts[s] = update_counts.get(s,0) + 1

      # next state becomes current state
      s = s2

    # log the reward for this episode
    reward_per_episode.append(episode_reward)

  plt.plot(reward_per_episode)
  plt.title("reward_per_episode")
  plt.show()

  """
  4. determine the policy from Q*
  ------------------------------
  - find V* from Q*

  this is the time that we build the policy, based on the Q-table.
  """

  policy = {}
  V = {}
  for s in grid.actions.keys(): # loop over all the state
    a, max_q = max_dict(Q[s]) # for each state, take from the Q table to best action and its value
    policy[s] = a # assign the policy
    V[s] = max_q  # assign the value


  """
  5. Inspect the results
  ----------------------

  """
  # what's the proportion of time we spend updating each part of Q?
  print("update counts:")
  total = np.sum(list(update_counts.values()))
  for k, v in update_counts.items():
    update_counts[k] = float(v) / total
  print_values(update_counts, grid)

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)

