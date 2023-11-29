from __future__ import print_function, division
from builtins import range
import numpy as np
from grid_world import windy_grid, ACTION_SPACE


"""
Policy Iterative Evaluation 
============================
- What is the value of a given policy ?
In this code, we are going to evaluate a policy using dynamic programing 
"""

SMALL_ENOUGH = 1e-3 # threshold for convergence


def print_values(V, g):
  for i in range(g.rows):
    print("---------------------------")
    for j in range(g.cols):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  for i in range(g.rows):
    print("---------------------------")
    for j in range(g.cols):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")



if __name__ == '__main__':

  ### define transition probabilities and grid ###
  
  # the key is (s, a, s'), the value is the probability
  # that is, transition_probs[(s, a, s')] = p(s' | s, a)
  # any key NOT present will considered to be impossible (i.e. probability 0)
  # we can take this from the grid object and convert it to the format we want
  transition_probs = {}

  # to reduce the dimensionality of the dictionary, we'll use deterministic
  # rewards, r(s, a, s')
  # note: you could make it simpler by using r(s') since the reward doesn't
  # actually depend on (s, a)
  rewards = {}
  
  """
  1. define transition probabilities 
  ------------------------------------

  """
  # get the initialized windy grid 
  grid = windy_grid() 
  # Loop over the probability items:
  # its look like this -> ((2, 0), 'U'): {(1, 0): 1.0}
  # so we get the (state, action) and the value dict. 
  for (s, a), v in grid.probs.items(): 
    # from the value dict, extract the next state and its probability.
    # s2 -> the state
    # p  -> its probability
    for s2, p in v.items():
      # assign the value p to its corresponding state transition.
      # in current state `s` with the action `a`, this is the the probability `p` to pass the next state `s2`
      transition_probs[(s, a, s2)] = p
      # grab the corresponding reward, if there is. 
      # "if I pass from current state with some action to the next state, what is the reward"?
      rewards[(s, a, s2)] = grid.rewards.get(s2, 0)
      
  print("The rewards after define transition probabilities")
  # print(rewards)
  print(transition_probs)
  

  """
  2. Define probabilistic policy
  ------------------------------
  Now we defined a probabilistic policy!
  its mean that given a state, there is probability to make some action. 
  * e.g: if the current state is (2,0) - the probability to go 'U' is 50%, 
       and the probability to go 'R' is 50%.
  """ 
  # now, the policy is NOT deterministic, now is probabilistic!
  policy = {
    (2, 0): {'U': 0.5, 'R': 0.5}, # the current state, the probability to make actions.
    (1, 0): {'U': 1.0},
    (0, 0): {'R': 1.0},
    (0, 1): {'R': 1.0},
    (0, 2): {'R': 1.0},
    (1, 2): {'U': 1.0},
    (2, 1): {'R': 1.0},
    (2, 2): {'U': 1.0},
    (2, 3): {'L': 1.0},
  }
  print_policy(policy, grid)

  
  """
  3. initialize V(s) = 0
  -----------------------
  First, we initialize the Value (reward) of each state to zero. 
  """
  V = {}                      # inititlize Value dictionary
  for s in grid.all_states(): # loop over all the states in the enviroment
    V[s] = 0                  # initialize the value of each state to 0

  gamma = 0.9 # discount factor


  """
  4. Implement the Belman exquation for Policy evaluation
  --------------------------------------------------------
  
  Now, we are going to implement the Bellman equation for evaluate the
  policy for each state.
  
  - the value of each state is the cumelative reward from this state and so forth. 

  """

  # repeat until convergence
  it = 0
  while True:
    biggest_change = 0
    # 1. Loop for each state
    for s in grid.all_states(): 
      # If the game is not over
      if not grid.is_terminal(s): 
        # store the value of the state
        old_v = V[s]
        # we will accumulate the answer 
        new_v = 0 
        # 2. you are in a state, now loop over all the actions space
        for a in ACTION_SPACE: 
          # 3. you are in state `s` with action `a`, loop over all the states `s2` for calculate the cumelative reward.
          for s2 in grid.all_states(): # loop over the states
            # get the probability of doing action `a` in state `s`
            action_prob = policy[s].get(a, 0) # float
            # reward is a function of (s, a, s'), 0 if not specified
            # from the current state, I make action `s` and now the state is `s2`, give me the reward
            r = rewards.get((s, a, s2), 0)
            # Update the value
            new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + gamma * V[s2])

        # after done getting the new value, update the value table 
        # we'll update the Value for each state! 
        V[s] = new_v
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    print("iter:", it, "biggest_change:", biggest_change)
    print_values(V, grid)
    it += 1

    if biggest_change < SMALL_ENOUGH:
      break
  print("V:", V)
  print("\n\n")

  # sanity check
  # at state (1, 2), value is 0.5 * 0.9 * 1 + 0.5 * (-1) = -0.05

