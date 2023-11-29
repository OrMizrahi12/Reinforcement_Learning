from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid

# ----------------- helper functions -------------------

def max_dict(d): 
  # returns the argmax (key) and max (value) from a dictionary.
  # put this into a function since we are using it so often.

  # find max val
  max_val = max(d.values())

  # find keys corresponding to max val.
  # find which keys correspond to this maximum value.
  # so `max_keys` will be a list containing all the actions that yield the maximum value max value.
  max_keys = [key for key, val in d.items() if val == max_val]
  
  # return the max value, and a random optimal action from `max_keys`.  
  return np.random.choice(max_keys), max_val

def print_values(V, g):
  """
  A function that ptint the values.
  This function will draw our grid in ASCII, and inside each state 
  It will print the value corresponding to that state.
  
  Parameters
  ----------
  - `V`: A value dictionary
  - `g`: Grid 

  """ 
  
  for i in range(g.rows): # Loop over the grid rows
    print("---------------------------")
    for j in range(g.cols): # Loop over the columns in each row
      # Get the value for the current position [i,j]
      # we using default value of 0 (if we dont get any value, we get 0.)
      v = V.get((i,j), 0) 
      # check if the value is not negative (for print in appropirate formate)
      if v >= 0: 
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  """
  Very similar to the `print_values()` function. 
  This function is print the policy of our agant in the grid.
  in other words, in each state in the grid, what will be the policy?
  
  Parameters
  ----------
  - `P`: the policy (which action to do in this state?)
  - `g`: the grid world (the env)
  """

  for i in range(g.rows): # Loop over the grid rows
    print("---------------------------")
    for j in range(g.cols): # loop over the grig columns
      # given the state (i,j - location on the grid), give the the action based on the policy
      a = P.get((i,j), ' ')  
      print("  %s  |" % a, end="")
    print("")


# ----------------- End helper functions -------------------


"""
SARSA
======

> SARSA solve the control problem (find a well policy for a state)

Intro
-----
* SARSA builds the policy while interacting with the environment. 
* SARSA is an on-policy RL algorithm used for learning a policy for a Markov Decision Process (MDP). 
  The name "SARSA" stands for the key components of the algorithm: 
  State, Action, Reward, State', Action'. 
  It's a model-free method, 
  meaning it doesn't require knowledge of the underlying dynamics of the environment.

Components:
-----------
* State (S):   The current situation or configuration of the environment.
* Action (A):  The decision or move made in a given state.
* Reward (R):  The immediate feedback after taking an action in a particular state.
* State (S'):  The next state reached after taking an action in the current state.
* Action (A'): The next action chosen in the new state.

Formula:
--------
The SARSA algorithm uses the information gathered from one time step to 
update its current estimate of the value of the current state-action pair 
and to make decisions about the next action. 
The update rule for SARSA is as follows:

>>> Q(S,A) ← Q(S,A) + a * [R + y * Q(S',A') - Q(S,A)]

- Where:
  - Q(S,A)   -> The current estimate of the value of the state-action pair.
  - a        -> The learning rate, determining the step size of the update.
  - R        -> The immediate reward received after taking action A in state S 
  - y        -> The discount factor, representing the importance of future rewards. It's a value between 0 and 1.
  - Q(S',A') -> The estimated value of the next state-action pair.

  
Conclusion
----------
  
* SARSA follows an epsilon-greedy policy during action selection. 
* The SARSA algorithm is particularly useful in scenarios where the agent 
  needs to learn a policy for interacting with an environment while experiencing 
  the consequences of its actions in real-time. 
  It's considered an on-policy algorithm because 
  it learns the value of the policy that it is currently following.

"""



GAMMA = 0.9 # The disccount factor (the importance of future rewards)
ALPHA = 0.1 # learning rate (step size of the update)
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R') # action space


def epsilon_greedy(Q, s, eps=0.1):
  if np.random.random() < eps:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)
  else:
    a_opt = max_dict(Q[s])[0]
    return a_opt


if __name__ == '__main__':
  
  """
  1. Create an envirement
  """
  # grid = standard_grid()
  grid = negative_grid(step_cost=-0.1)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  """
  2. initialize Q(s,a) -> state,action.
  --------------------------------------

  the goal is to learn by experiment and "build" the policy.
  so later we are going to update the Q(s,a).
  """
  
  Q = {}
  states = grid.all_states() # get all the state
  for s in states:
    Q[s] = {} # initialize each state
    for a in ALL_POSSIBLE_ACTIONS: # loop over the actions
      Q[s][a] = 0 # in each state, initialize all the posible actions.

  # let's also keep track of how many times Q[s] has been updated
  update_counts = {}
  
  """
  3. SARSA implementation
  ---------
  >>> repeat until convergence
  """
  
  # initialize reward per episode
  # If we see this increase over time, 
  # that's a good sign that our agents is learning.
  reward_per_episode = []
  
  # loop for 10000 episodes
  for it in range(10000):
    if it % 2000 == 0:
      print("it:", it)

    # begin a new episode - reset the env, start a new game.
    s = grid.reset()

    # take an action given a state
    a = epsilon_greedy(Q, s, eps=0.1)

    # for cummelate all the rewards we rechived
    episode_reward = 0

    # while the game is not over...
    while not grid.game_over():
      
      # perform action and get next state + reward
      r = grid.move(a)          # reward
      s2 = grid.current_state() # state

      # update cumelated reward
      episode_reward += r

      # get next action
      a2 = epsilon_greedy(Q, s2, eps=0.1)

      # update Q(s,a)
      Q[s][a] = Q[s][a] + ALPHA*(r + GAMMA*Q[s2][a2] - Q[s][a])

      # we would like to know how often Q(s) has been updated too
      update_counts[s] = update_counts.get(s,0) + 1

      # next state becomes current state
      s = s2 # update state
      a = a2 # update action

    # log the reward for this episode
    reward_per_episode.append(episode_reward)

  plt.plot(reward_per_episode)
  plt.title("reward_per_episode")
  plt.show()
  
  """
  4. Use Q(s,a) to find the optimal policy and the optimal state
  ---------------------------------------------------------------
  
  now that we have the Q(s,a) (state-avtion), 
  we can determine the policy.
  
  - determine the policy from Q*
  - find V* from Q*

  """
  
  policy = {} # initialize policy dicy
  V = {}      # initialize Values dict (V(s))
  
  # Loop over all the states
  for s in grid.actions.keys():
    # given a state -> get the action and and its corresponsed max value.   
    a, max_q = max_dict(Q[s])
    # for this state, assign the policy with the best action
    policy[s] = a
    # assign the value of this state
    V[s] = max_q

  """
  5. Inspect the results
  ------------------------

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

