from __future__ import print_function, division
from builtins import range
import numpy as np
from grid_world import standard_grid, negative_grid

GAMMA = 0.9


"""

Solve the prediction problem using Monte Carlo

"""


# NOTE: this is only policy evaluation, not optimization


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




def play_game(grid, policy, max_steps=20):
  """
  returns a list of states and corresponding returns.
  """
 
  # reset game to start at a random position.
  # we need to do this, because given our current deterministic policy.
  # we would never end up at certain states, but we still want to measure their value.
  start_states = list(grid.actions.keys())        # take the state list
  start_idx = np.random.choice(len(start_states)) # random choise state
  grid.set_state(start_states[start_idx])         # set the random state

  s = grid.current_state()

  # keep track of all states and rewards encountered
  states = [s]  # list for store the states
  rewards = [0] # list for store the rewards

  steps = 0
  while not grid.game_over():
    a = policy[s]    # given a state, take the action based on the policy  
    r = grid.move(a) # move (apply the action), and take the reward
    next_s = grid.current_state() # store the next state

    # update states and rewards lists
    states.append(next_s)
    rewards.append(r)

    steps += 1
    if steps >= max_steps:
      break

    # update state
    # note: there is no need to store the final terminal state
    s = next_s

  # we want to return:
  # states  = [s(0), s(1), ..., S(T)]
  # rewards = [R(0), R(1), ..., R(T)]

  return states, rewards


if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = standard_grid() # initialize the frid obj

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  policy = {
    (2, 0): 'U', # state -> action
    (1, 0): 'U', # state -> action
    (0, 0): 'R', 
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U', 
  }

  # initialize V(s) and returns
  V = {}
  returns = {} # dictionary of state -> list of returns we've received
  states = grid.all_states()
  for s in states:
    if s in grid.actions:
      returns[s] = []
    else:
      # terminal state or state we can't otherwise get to
      V[s] = 0

  # repeat (implement the Monte carlo)
  for _ in range(100):
    # generate an episode using pi
    # given the grid (env) and the policy (the act on the agent in a giveb state)
    # we generate states and their rewards.
    # the [states, rewards] generation is randomaly! ()
    states, rewards = play_game(grid, policy)
    # initialize G (expected return from sample)
    G = 0
    T = len(states)
    # we loop through our states and rewards backwards,
    # since the return is calculated recursively based on future returns.
    for t in range(T - 2, -1, -1): # T - 2 -> the last state before the terminal state.
      s = states[t]     # state in time t
      r = rewards[t+1]  # reward on t+1
      G = r + GAMMA * G # update return

      # we'll use first-visit Monte Carlo
      if s not in states[:t]: # if we still not sample this state (first visit MC)
        returns[s].append(G) # append the G (value) to the list of the returns[in the given state] 
        V[s] = np.mean(returns[s]) # take the mean of expected return in the state (V[on the given state]). 

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)
