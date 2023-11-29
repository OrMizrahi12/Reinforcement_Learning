from __future__ import print_function, division
from builtins import range
import numpy as np
from grid_world import standard_grid, ACTION_SPACE
from iterative_policy_evaluation_deterministic import print_values, print_policy




"""
Policy improvment 
=================

We are going to see how we can improve a policy. 
"""

SMALL_ENOUGH = 1e-3
GAMMA = 0.9


"""
1. Get transition probabilities and rewards
-------------------------------------------

this function going to setup the 

transition probabilities (tuple):
  - its the probability to pass from state to the next state with some given action. 

rewards (tuple):
  - store the state, action and its corresponding reward in a dict  
"""
# copied from iterative_policy_evaluation
def get_transition_probs_and_rewards(grid):
  """
  define transition probabilities and rewards
  - the key is (s, a, s'), the value is the probability
  - that is, transition_probs[(s, a, s')] = p(s' | s, a)
  - any key NOT present will considered to be impossible (i.e. probability 0)
  """

  # For stire the transition probabilities
  transition_probs = {}

  # to reduce the dimensionality of the dictionary, we'll use deterministic
  # rewards, r(s, a, s')
  # note: you could make it simpler by using r(s') since the reward doesn't
  # actually depend on (s, a)
  rewards = {}

  # 1. for each row
  for i in range(grid.rows): 
    # 2. for each column in the row
    for j in range(grid.cols): #
      # take the current state (location)
      s = (i, j) 
      # if the game is not over
      if not grid.is_terminal(s):
        # now you in a state (specific location), now loop over all the actions 
        for a in ACTION_SPACE: 
          # pass the current state and the action, get the next state
          s2 = grid.get_next_state(s, a) 
          # set the probability to move from state `s` to next state `s2` with action `a` to - 1
          transition_probs[(s, a, s2)] = 1
          # If there is a reward to that state 
          if s2 in grid.rewards: 
            # assign the reward dict
            rewards[(s, a, s2)] = grid.rewards[s2] 
  
  # return the transition probability and the reward
  return transition_probs, rewards


def evaluate_deterministic_policy(grid, policy, initV=None):
  
  """
  2. Evaluate a deterministic policy.
  ------------------------------------

  Its somting we already seen, its evaluate the policy of of each state ( V(s) )
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

  # repeat until convergence (implement Bellman equation)
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
3. Improve the policy! 
-----------------------

This part we've not seen before. 
we are going to improve the policy!

"""

if __name__ == '__main__':

  # 1. initialize the grid
  grid = standard_grid()
  
  # 2. get the transition probability dict and the reward dict. 
  transition_probs, rewards = get_transition_probs_and_rewards(grid)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # 3. Initialize ramdom policy (we are going to improve that later...)
  policy = {}  # state -> action
  # fill each state with random action
  for s in grid.actions.keys(): 
    # e.g: (1,2) -> 'U' 
    policy[s] = np.random.choice(ACTION_SPACE)

  # Print the random initial policy
  print("initial policy:")
  print_policy(policy, grid)

  
  """
  Improve the policy! 
  -------------------
  This code going to find the best policy for each state
  """
  # repeat until convergence - will break out when policy does not change
  V = None # value 
  while True:

    # policy evaluation step.
    # we evaluate the getting back the policy dictionarity
    # the V dictionarity that contain the evaluated policy of each state.
    V = evaluate_deterministic_policy(grid, policy, initV=V)
    
    # ...now after we have V, we want try improve that....

    # policy improvement step
    is_policy_converged = True

    # 1. Loop over states
    for s in grid.actions.keys():
      # store the action 
      old_a = policy[s] 
      new_a = None 
      best_value = float('-inf')
      # 2. You already in a specific state, now loop over all the actions! 
      #    we loop over all the action for find the best action for a state. (the best policy for a specific state.)
      for a in ACTION_SPACE:
        v = 0
        # 3. You in state `s` with action `a`, 
        # now loop over all the states,
        # (for calculate the cumelative reward from this state to all the possible states with this action)
        for s2 in grid.all_states():
          # reward is a function of (s, a, s'), 0 if not specified
          r = rewards.get((s, a, s2), 0)
          # update the value function
          v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])
         
        # you stil in the same state (loop 1), update the best policy for this spesific state! 
        # at the end, its going to contain the BEST action for this spesific state!
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
