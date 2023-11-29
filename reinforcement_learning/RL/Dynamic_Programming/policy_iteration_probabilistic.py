from __future__ import print_function, division
from builtins import range
import numpy as np
from grid_world import windy_grid, windy_grid_penalized, ACTION_SPACE
# from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9

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


# copied from iterative_policy_evaluation
def get_transition_probs_and_rewards(grid):
  
  """
  define transition probabilities and grid
  =========================================
  Generate two dictionaties:
  - Transition probability
    - this is a dictionary that contain all the probability of the transition 
      from state to state with an action. 
  - RewardL a dictionary that contaon the rewards and their corresponding state/action. 
  """
  
  # the key is (s, a, s'), the value is the probability
  # that is, transition_probs[(s, a, s')] = p(s' | s, a)
  # any key NOT present will considered to be impossible (i.e. probability 0)
  transition_probs = {}

  # to reduce the dimensionality of the dictionary, we'll use deterministic
  # rewards, r(s, a, s')
  # note: you could make it simpler by using r(s') since the reward doesn't
  # actually depend on (s, a)
  rewards = {}
  
  # 1. Loop for the (state,action), and the `v` (value tuple)
  for (s, a), v in grid.probs.items():
    # 2. Loop for the state and the provavility in the `v` (value tuple)
    for s2, p in v.items():
      # assign the probability to this transition
      transition_probs[(s, a, s2)] = p
      # assign the reward (if any)
      rewards[(s, a, s2)] = grid.rewards.get(s2, 0)

  return transition_probs, rewards


def evaluate_deterministic_policy(grid, policy, initV=None):
  """
  Function for evaluate the policy of each state ( V(s) )
  ========================================================

  This function is evalate the policy of each state.
  
  in other words - the best action for a given state
  """
  # initialize V(s) = 0
  if initV is None:
    V = {}
    for s in grid.all_states():
      V[s] = 0
  else:
    # it's faster to use the existing V(s) since the value won't change
    # that much from one policy to the next
    V = initV

  # repeat until convergence >>>
  it = 0
  while True:
    biggest_change = 0
    for s in grid.all_states():
      if not grid.is_terminal(s):
        old_v = V[s]
        new_v = 0 # we will accumulate the answer
        for a in ACTION_SPACE:
          for s2 in grid.all_states():

            # action probability is deterministic
            action_prob = 1 if policy.get(s) == a else 0
            
            # reward is a function of (s, a, s'), 0 if not specified
            r = rewards.get((s, a, s2), 0)
            new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

        # after done getting the new value, update the value table
        V[s] = new_v
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))
    it += 1

    if biggest_change < SMALL_ENOUGH:
      break
  return V


"""
Find the best policy
======================

In this code below, we are going to find the best policy for a state

in other words, find the best action for a given state!
"""

if __name__ == '__main__':
  
  # Grid with step cost (each step have cost.)
  grid = windy_grid_penalized(step_cost=-0.1)
  # grid = windy_grid()
  transition_probs, rewards = get_transition_probs_and_rewards(grid)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  # we'll randomly choose an action and update as we learn
  policy = {}
  for s in grid.actions.keys():
    policy[s] = np.random.choice(ACTION_SPACE)

  # initial policy
  print("initial policy:")
  print_policy(policy, grid)

  # repeat until convergence - will break out when policy does not change
  V = None
  while True:

    # policy evaluation step - we already know how to do this!
    V = evaluate_deterministic_policy(grid, policy, initV=V)

    # policy improvement step
    is_policy_converged = True
    for s in grid.actions.keys():
      old_a = policy[s]
      new_a = None
      best_value = float('-inf')

      # loop through all possible actions to find the best current action
      for a in ACTION_SPACE:
        v = 0
        for s2 in grid.all_states():
          # reward is a function of (s, a, s'), 0 if not specified
          r = rewards.get((s, a, s2), 0)
          v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

        if v > best_value:
          best_value = v
          new_a = a

      # new_a now represents the best action in this state
      policy[s] = new_a
      if new_a != old_a:
        is_policy_converged = False

    if is_policy_converged:
      break

  # once we're done, print the final policy and values
  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)
