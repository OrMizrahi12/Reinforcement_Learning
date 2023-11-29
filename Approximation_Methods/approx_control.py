from __future__ import print_function, division
from builtins import range
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from halper import print_values, print_policy
from sklearn.kernel_approximation import Nystroem, RBFSampler



"""
Approximation control 
=====================
* This code implements reinforcement learning with function approximation for control problems
*  the goal is to learn an optimal policy in order to maximize the expected cumulative reward
* The method used for function approximation is the Radial Basis Function (RBF) Sampler. 
"""


GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# mapping actions to integerse {'U': 0, 'D': 1, 'L': 2, 'R': 3} 
ACTION2INT = {a: i for i, a in enumerate(ALL_POSSIBLE_ACTIONS)} 

# the actions to one hot encoding
INT2ONEHOT = np.eye(len(ALL_POSSIBLE_ACTIONS)) 


def epsilon_greedy(model, s, eps=0.1):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps): # the best policy came from the model (explode)
    values = model.predict_all_actions(s)
    return ALL_POSSIBLE_ACTIONS[np.argmax(values)]
  else: # take a random (explore)
    return np.random.choice(ALL_POSSIBLE_ACTIONS)


def one_hot(k):
  return INT2ONEHOT[k]


def merge_state_action(s, a):
  """
  Merge the state and the action into an array 
  (we need an array beacuse we perform features transformation.)
  """
  # pass the integer (ACTION2INT[a]) to one hot  
  ai = one_hot(ACTION2INT[a]) 
  # concatenate the state with this one hot encoding of the action.
  # (1,3,0,0,0,1) -> (1,3) = state , (0,0,0,1) = one hot of this action 
  return np.concatenate((s, ai))  


def gather_samples(grid, n_episodes=1000):
  samples = []
  for _ in range(n_episodes):
    s = grid.reset()
    while not grid.game_over():
      a = np.random.choice(ALL_POSSIBLE_ACTIONS)
      sa = merge_state_action(s, a)
      samples.append(sa)

      r = grid.move(a)
      s = grid.current_state()
  return samples


class Model:
  def __init__(self, grid):
    """
    initializes a function approximation model using the `RBFSampler()` as the featurizer. 
    It fits the featurizer to the state samples obtained from `gather_samples()`,
    and initializes the weights (self.w) to zeros.
    """
    # fit the featurizer to data
    samples = gather_samples(grid) # get the samples
    # self.featurizer = Nystroem()
    self.featurizer = RBFSampler() # create an instance of the featurizer 
    self.featurizer.fit(samples)   # create the features based on the samples
    dims = self.featurizer.n_components # take the number of features dumentionalites

    # initialize linear model weights
    self.w = np.zeros(dims)
 
  def predict(self, s, a):
    """
    A function to predict the value of Q(s,a) of one [state,action]
    """
    sa = merge_state_action(s, a)          # create an array of [state,action]
    x = self.featurizer.transform([sa])[0] # based on the, create features 
    return x @ self.w # take the dot product between the features and the model weigths
  
  def predict_all_actions(self, s):
    """
    Function to predict all the Q(s,a)
    - this loops through all possible actions, given a status `s`,
      and computes the prediction for each state action pair.
    """
    return [self.predict(s, a) for a in ALL_POSSIBLE_ACTIONS]

  def grad(self, s, a):
    """
    return featurizersed Q(s,a)
    """
    sa = merge_state_action(s, a)
    x = self.featurizer.transform([sa])[0]
    return x
  
"""
>>> Implement the approximation method for control problem 
"""
if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  # grid = standard_grid()
  grid = negative_grid(step_cost=-0.1) # initial grid env

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)
  
  
  model = Model(grid)     # initial model
  reward_per_episode = [] # list for store the reward of each epicode
  state_visit_count = {}  # store the number of times that we visit in each state 

  # repeat until convergence
  n_episodes = 20000
  for it in range(n_episodes):
    if (it + 1) % 100 == 0:
      print(it + 1)

    s = grid.reset() # reset the env
    state_visit_count[s] = state_visit_count.get(s, 0) + 1 # update the count the we visit in each state 
    episode_reward = 0 # initial epicode reward   
    while not grid.game_over():
      a = epsilon_greedy(model, s) # based on the state and the model, get an action
      r = grid.move(a)             # perform the action
      s2 = grid.current_state()    # get the current state
      state_visit_count[s2] = state_visit_count.get(s2, 0) + 1 # update the number of visits in this state

      ## update the target (future rewards) ##
      if grid.game_over():
        target = r
      else:
        values = model.predict_all_actions(s2) # predict the value, giveb the state
        # is the reward + the max value (the action that yield the best value from all the action, in this state.)
        target = r + GAMMA * np.max(values)    # update the value 

      ## update the model ## 
      g = model.grad(s, a)               # create feature based on the state and action
      err = target - model.predict(s, a) # update the err
      model.w += ALPHA * err * g         # update the model weigths 
      
      # update accumulate reward
      episode_reward += r

      # update state
      s = s2
    
    reward_per_episode.append(episode_reward)

  plt.plot(reward_per_episode)
  plt.title("Reward per episode")
  plt.show()

  # obtain V* and pi*
  V = {}             # Dictionary for store the best values
  greedy_policy = {} # Dictionary for store the best policy
  states = grid.all_states() # get all the states
  
  # loop over all the states
  for s in states: 
    if s in grid.actions: # is the state in the actions -->  
      values = model.predict_all_actions(s) # predict the values, based on the state
      V[s] = np.max(values) # store the value that came from the best action for this state
      greedy_policy[s] = ALL_POSSIBLE_ACTIONS[np.argmax(values)] # save this action that yield the best value  
    else:
      # terminal state or state we can't otherwise get to
      V[s] = 0

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(greedy_policy, grid)


  print("state_visit_count:")
  state_sample_count_arr = np.zeros((grid.rows, grid.cols))
  for i in range(grid.rows):
    for j in range(grid.cols):
      if (i, j) in state_visit_count:
        state_sample_count_arr[i,j] = state_visit_count[(i, j)]
  df = pd.DataFrame(state_sample_count_arr)
  print(df)
