from __future__ import print_function, division
from builtins import range
import numpy as np
from grid_world import windy_grid, ACTION_SPACE
from policy_iteration_deterministic import print_values, print_policy


"""
Value iteration
================

The value iteration is more efficient way for find the best policy. 
"""

# this variable is for track after the change improvment of the policy.
# if the policy[i] - the policy[i-1] < 1e-3, we'll stop the loop. 
SMALL_ENOUGH = 1e-3
# define learning rate
GAMMA = 0.9

# copied from iterative_policy_evaluation
def get_transition_probs_and_rewards(grid):
  """
  define transition probabilities and grid
  ==========================================
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

  for (s, a), v in grid.probs.items():
    for s2, p in v.items():
      transition_probs[(s, a, s2)] = p
      rewards[(s, a, s2)] = grid.rewards.get(s2, 0)

  return transition_probs, rewards



"""
Value iteration code below
==========================

"""


if __name__ == '__main__':
  grid = windy_grid()
  transition_probs, rewards = get_transition_probs_and_rewards(grid)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # initialize V(s)
  V = {}
  states = grid.all_states()
  for s in states:
    V[s] = 0

  
  """
  1. Value iteration loop
  -------------------------

  Find the best value of each state! (the optimal value function)
  """
  # repeat until convergence
  # V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
  it = 0
  while True: 
    biggest_change = 0
    # 1. Loop over all the state
    for s in grid.all_states():
      if not grid.is_terminal(s):
        old_v = V[s]
        new_v = float('-inf')
        # 2. Loop over all the actions space
        for a in ACTION_SPACE:
          v = 0
          # 3. Loop over all the state again 
          for s2 in grid.all_states():
            # reward is a function of (s, a, s'), 0 if not specified
            r = rewards.get((s, a, s2), 0)
            v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

          # keep v if it's better
          if v > new_v:
            new_v = v

        V[s] = new_v
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))
    
    # Brake the while loop if the change is small enough
    it += 1
    if biggest_change < SMALL_ENOUGH:
      break

  """
  2. find a policy that leads to optimal value function
  -----------------------------------------------------

  The next step is to find the optimal policy by taking the argmax over the value function.
  """
  policy = {}
  for s in grid.actions.keys():
    best_a = None
    best_value = float('-inf')
    # loop through all possible actions to find the best current action
    for a in ACTION_SPACE:
      v = 0
      for s2 in grid.all_states():
        # reward is a function of (s, a, s'), 0 if not specified
        r = rewards.get((s, a, s2), 0)
        v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

      # best_a is the action associated with best_value
      if v > best_value:
        best_value = v
        best_a = a
    policy[s] = best_a

  # our goal here is to verify that we get the same answer as with policy iteration
  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)
