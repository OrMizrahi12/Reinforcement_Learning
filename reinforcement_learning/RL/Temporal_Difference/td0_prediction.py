from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid


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




"""
Temporal Difference (0) prediction
======================================

> Temporal Difference (0) for solve the prediction problem (predict the V(s)). 

Intro
------

* TD(0), which stands for Temporal Difference with a lookahead of 0 steps,
  is a type of model-free prediction algorithm that estimates the value function of a 
  policy based on the observed transitions and rewards. 

key concepts:
-------------

1. Temporal Difference (TD):
  * TD learning is a type of reinforcement learning algorithm that 
    combines aspects of dynamic programming and Monte Carlo methods.
  * it updates the estimate for a state based on the difference between the current estimate and a 
    bootstrapped estimate of the value of the next state.

2. Prediction
  * prediction refers to estimating the expected future rewards or values associated with different states.  

3. TD(0) Prediction: 
  * TD(0) is a specific instance of TD learning where the lookahead is 0 steps.
  * it updates the value of a state based solely on the immediate reward and the estimated value of the next state.

  * V(S[t]) ← V(S[t]) + a * [R[t+1] + y * V(S[t+1]) - V(S[t])]
    Where: 
    - V(S[t]) -> is the estimated value of S[t]
    - R[t+1]  -> is the immediate reward  after taking an action in state S[t]
    - y       -> is the discount factor, representing the importance of future rewards.
    - a       -> is the learning rate, determining the size of the update

Conclusion
-----------
TD(0) is a sample-based update method, 
meaning it updates the value function after each time step based on the observed transition. 
It has the advantage of being computationally less expensive than full Monte Carlo methods, 
as it doesn't require waiting until the end of an episode to make updates. 
However, it may have higher variance in its estimates compared to methods 
that look further into the future (e.g., TD with larger lookahead or Monte Carlo methods).

"""



SMALL_ENOUGH = 1e-3 # treshold for small enough change
GAMMA = 0.9         # gamma factor: representing the importance of future rewards.
ALPHA = 0.1         # is the learning rate, determining the size of the update.
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R') # action space


def epsilon_greedy(policy, s, eps=0.1):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps):
    return policy[s]
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)


if __name__ == '__main__':
  
  """
  1. Initialize grid env
  ----------------------
  """
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = standard_grid()

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)
  
  """
  2. define policy
  -----------------
  """
  policy = {
    (2, 0): 'U', # state -> action
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
  }

  """
  3. # initialize V(s) and returns
  ---------------------------------
  """
  V = {}
  states = grid.all_states()
  for s in states:
    V[s] = 0
  
  """
  4. create an empty list to store a Deltas for each episode
  -----------------------------------------------------------
  
  - store max change in V(s) per episode

  """
  deltas = []

  """
  5. Uplay the TD(0)
  ------------------

  """
  # repeat until convergence
  n_episodes = 10000
  for it in range(n_episodes):
    # begin a new episode
    s = grid.reset()
    delta = 0

    # if the game not over, loop >>>
    while not grid.game_over():
      
      # given policy and state, rechive an action 
      a = epsilon_greedy(policy, s)
      
      # aplay the action, get back the reward
      r = grid.move(a)

      # now take the state
      s_next = grid.current_state()

      # store the old
      v_old = V[s] 
      # TD(0) - update V(s) - the value of a specific state.  
      V[s] = V[s] + ALPHA*(r + GAMMA*V[s_next] - V[s])
      
      # get the delta - the max change between V(s) and old_v(3)  
      # its use for see the progras later...
      delta = max(delta, np.abs(V[s] - v_old))
      
      # next state becomes current state
      s = s_next

    # store delta (for plot the prograss later..)
    deltas.append(delta)



  """
  6. Inspect the results
  ----------------------
  """
  plt.plot(deltas)
  plt.show()

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)



