from __future__ import print_function, division
from builtins import range
import numpy as np
from grid_world import standard_grid, ACTION_SPACE




"""
Policy Iterative Evaluation 
============================
- What is the value of a given policy ?
In this code, we are going to evaluate a policy using dynamic programing 
"""



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

  """
  define transition probabilities and grid
  """

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

  """
  1. Define transition probabilities
  -------------------------------

  Here, we define the transition probability. what is mean?
  We'll get a map of all the states, the actions, and the corresponsed reward and states
  
  example of one element in the `transition_probs`:
    -  ((0, 0), 'R', (0, 1)): 1 
      - (0, 0)     -> the current state
      - 'R'        -> the action 
      - (0, 1)): 1 -> the next state and its probability
  
  About the `rewards`:
    - the reward will contain all the action and state that there is rewards.
    - example:
    - ((0, 2), 'R', (0, 3)): 1
      - (0, 2), 'R' -> the current state and the action. 
      - (0, 3)): 1  -> the next state and its reward.
  """  
  # initialize & get the grid
  grid = standard_grid()     
  # loop over the grid row      
  for i in range(grid.rows): 
    # in each row, loop over the grid columns
    for j in range(grid.cols):
      # set the current state (tuple, the current location on the grid.)
      s = (i, j) 
      # if the game is not over
      if not grid.is_terminal(s): 
        # now you are in a specific location in the grid (in a specific state).
        # at this point. loop over all the actions
        for a in ACTION_SPACE: # loop over the action space (in this case: Up,Down,Left,Rigth)
          # based on the current state and the action, give me the next state.
          # s2 -> is a tuple of (i,j) -> the next state
          s2 = grid.get_next_state(s, a)   
          # from current state `s` with action `a` to next state `s2` --> set the probability to 1.
          transition_probs[(s, a, s2)] = 1  
          # check if the next state in the rewards dictionary attribute of the grid world object.
          # if there is a reward in the state, store it in a rewards dictionary.
          if s2 in grid.rewards: 
            # assign this reward to our rewards dictionary
            # [(s, a, s2) -> the reward for go from state `s` with action `a` to state `s2`. 
            rewards[(s, a, s2)] = grid.rewards[s2] 
  
  print(transition_probs)
  print(rewards)

  """
  2. Define a fixed policy 
  ------------------------

  the key -> state
  the value -> action
   
  given a state, tell me which action to do. 
    - e.g: you in the location (2, 0), go up 'U'.  
  """
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
  """
  3. Initialize V(s)
  -------------------
  A dictionary V is initialized to store the values of states. Initially, all values are set to 0.

  - The V(s) refers to the expected cumulative reward that an agent can obtain 
    from that state onward while following a specific policy.
  """
  V = {}                      # create dictionary
  for s in grid.all_states(): # we loop over all the states
    V[s] = 0                  # initialize

  gamma = 0.9 # discount factor
  
  """
  4. The policy evaluation code
  -----------------------------
  > This is the "meat" of this script where we actually implement the algorithm we've been discussing.

  - So now, we are going to enters a loop for policy evaluation,
    which iteratively updates the estimated values of states until convergence.

  So what this function are doing? 
    - This function evaluate the policy for each state!
      - For each state what going to by the future reward ?!

  """
   
  
 
  # repeat until convergence (this is the belman equation implementation)
  it = 0
  while True:
    biggest_change = 0
    # 1. Loop over all the states
    for s in grid.all_states(): 
      # 2. check if the game is noe over
      if not grid.is_terminal(s):  
        old_v = V[s] # store the value given the state
        new_v = 0    # we will accumulate the answer
        # 3. you are in a specific state, now loop over the action space
        for a in ACTION_SPACE: 
          # 4. now you in state `s`, with action `a`. 
          # loop over the states again, for calculate the value from your current state to all the another
          # states with a given action.     
          for s2 in grid.all_states(): 
            # action probability is deterministic:
            # if the action that came from the policy given state `s` == to the action `a`, we set the probability to 1. 
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
