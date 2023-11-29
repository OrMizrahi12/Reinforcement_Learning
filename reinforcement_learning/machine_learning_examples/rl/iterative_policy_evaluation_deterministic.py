from __future__ import print_function, division
from builtins import range
import numpy as np
from grid_world import standard_grid, ACTION_SPACE


# threshold for convergence
# where the differences between the current and the previuos step is 
# small, we want to finish the code beacuse there is no improvment or change.
SMALL_ENOUGH = 1e-3 



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



if __name__ == '__main__':

  ### define transition probabilities and grid ###

  # the key is -> (s, a, s'), 
  # the value is -> the probability.
  # that is, transition_probs[(s, a, s')] = p(s' | s, a)
  # any key NOT present will considered to be impossible (i.e. probability 0)
  # (What is the probability that we transit from start to another?)
  transition_probs = {}

  # to reduce the dimensionality of the dictionary, we'll use deterministic
  # rewards, r(s, a, s')
  # note: you could make it simpler by using r(s') since the reward doesn't
  # actually depend on (s, a)
  # the key is -> the state,action, and the next state
  # the value is -> the corresponding reward! 
  rewards = {}

  ## Define transition probabilities ## 
  grid = standard_grid() # initialize the grid & return grid object
  for i in range(grid.rows): # loop over the grid row
    for j in range(grid.cols): # in eahc row, loop over the grid columns
      s = (i, j) # set the current state (tuple, the current location on the grid.)
      if not grid.is_terminal(s): # cheack if the state is not the end of the game
        for a in ACTION_SPACE: # loop over the action space (in this case: Up,Down,Left,Rigth)
          s2 = grid.get_next_state(s, a) # based on the current state and the action, give me the next state
          transition_probs[(s, a, s2)] = 1 # initialize 
          if s2 in grid.rewards: # check if the next state in the rewards dictionary attribute of the grid world object. 
            rewards[(s, a, s2)] = grid.rewards[s2] # assign this reward to our rewards dictionary 

  ## fixed policy
  # key -> state
  # value -> action
  policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'U',
    (2, 1): 'R',
    (2, 2): 'U',
    (2, 3): 'L',
  }
  print_policy(policy, grid)

  # initialize V(s) = 0 (this is the value function)
  V = {}
  for s in grid.all_states(): # we loop over all the states
    V[s] = 0 # initialize all the values to be 0 in any state.

  gamma = 0.9 # discount factor
  
  ## The policy evaluation code 
  # This is the "meat" of this script where we actually implement the algorithm we've been discussing.
 
  # repeat until convergence (this is the belman equation implementation)
  it = 0
  while True:
    biggest_change = 0
    for s in grid.all_states(): # Loop iver all the states
      if not grid.is_terminal(s): # chack if the state is not terminal (if the game not over) 
        old_v = V[s] # store the value given the state
        new_v = 0 # we will accumulate the answer
        for a in ACTION_SPACE: # loop in the action space
          # loop again in the states. think about the intuativitly:
          # you are already in a state, and you now hold in an action.
          # so you want again see all the states for know what to do.  
          for s2 in grid.all_states(): 

            # action probability is deterministic.
            # so if the policy given the state is equalt to the action,
            # we set the action probability to 1.
            action_prob = 1 if policy.get(s) == a else 0
            
            # reward is a function of (s, a, s'), 0 if not specified
            r = rewards.get((s, a, s2), 0)
            # cummelate the result (V)
            new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + gamma * V[s2])

        # after done getting the new value, update the value table
        V[s] = new_v
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    print("iter:", it, "biggest_change:", biggest_change)
    print_values(V, grid)
    it += 1

    if biggest_change < SMALL_ENOUGH:
      break
  print("\n\n")
