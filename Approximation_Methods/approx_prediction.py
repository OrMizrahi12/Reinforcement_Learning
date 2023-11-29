from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from halper import print_values, print_policy
from sklearn.kernel_approximation import Nystroem, RBFSampler



"""
Approximation method
=====================

this implements a RL algorithm using function approximation 
(in this case, using radial basis function features) to estimate the value function for a given policy.

"""

GAMMA = 0.9
ALPHA = 0.01
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


## Epsilon-Greedy Exploration
def epsilon_greedy(greedy, s, eps=0.1):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps):
    return greedy[s]
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)


## Data Collection:
def gather_samples(grid, n_episodes=10000):
  """
  The `gather_samples` function collects state samples by simulating episodes 
  in the grid-world environment. It starts from a random initial state, 
  takes random actions, and records the resulting states until the episode ends.
  """
  samples = []
  # 1. Loop for n episodes
  for _ in range(n_episodes):
    s = grid.reset()  # reset env (get initial state)
    samples.append(s) # push initial state to the samples list
    # 2. While the game is not over
    while not grid.game_over():
      a = np.random.choice(ALL_POSSIBLE_ACTIONS) # choose a random action 
      r = grid.move(a)                           # perform the action
      s = grid.current_state()                   # get current state
      samples.append(s)                          # append the current state in the samples list

  # return the samples   
  return samples


## Function Approximation Model:
class Model:
  def __init__(self, grid):
    """
    Linear function approximation model using radial basis function features.
    ------
    the init method initializes the featurizer with RBFSampler 
    and the weight vector (self.w) with zeros.
    
    """
    
    # take the samples 
    # it helps the model to understand the distribution of states 
    # that it is likely to encounter during training 
    samples = gather_samples(grid)
    
    # try that
    # self.featurizer = Nystroem()
    
    # Create the featurizer object. 
    # this featurizer will create features based on the samples (samples = list of the states.)
    # so the featurizer'll create features that relate to the states (maybe about their location..) 
    self.featurizer = RBFSampler() 
  
    # fit the featurizer to data
    self.featurizer.fit(samples)

    # take the dimensionalities of the features
    dims = self.featurizer.n_components

    # initialize linear model weights
    self.w = np.zeros(dims)
  
  def predict(self, s):
    """
     predicts the value of a state by applying the featurizer to the state 
     and computing the dot product with the weight vector.

     - it take state `s` as input, transform it to a feature, and predict the V(s).
    """
    # create the features based on the state
    x = self.featurizer.transform([s])[0]
    # predic by dot product between:
    # x      => the features
    # self.w => the model weigths 
    return x @ self.w 
  
  def grad(self, s):
    """
    returns the feature vector for a given state, 
    which is used in the gradient descent update.
    """
    # create the features based on the state 
    x = self.featurizer.transform([s])[0]
    return x # return that


## Main Training Loop:
if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = standard_grid()

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  greedy_policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
  }

  # Create a model, and pass the grid object. 
  model = Model(grid)
  mse_per_episode = []

  # repeat until convergence
  """
  Purpose
  ---------
  * To demonstrate demonstrate a simple reinforcement learning setup using function
    approximation (specifically, radial basis functions) to estimate the value function.
  * The agent interacts with a grid-world environment, and the function approximation 
    model is updated to approximate the values associated with the provided greedy policy.
  * The MSE per episode is used to monitor the learning progress.
  * The ultimate goal is to see how well the function approximation 
    captures the values in the grid-world environment.
  """

  
  n_episodes = 10000
  for it in range(n_episodes):
    if (it + 1) % 100 == 0:
      print(it + 1)
   
    s = grid.reset()      # reset the envirement
    Vs = model.predict(s) # predict the V(s) by the model
    n_steps = 0           # value for measure the steps in an epicode
    episode_err = 0       # for neasure the error in an epicode
    
    # ...While the game not over... 
    while not grid.game_over():
      a = epsilon_greedy(greedy_policy, s) # given policy and state, return an action
      r = grid.move(a)                     # perform tha action, take reward
      s2 = grid.current_state()            # store this current state

      # compute the target
      if grid.is_terminal(s2):
        target = r
      else:
        Vs2 = model.predict(s2)  # predict the value of the next state
        target = r + GAMMA * Vs2 # update the target (update the V(s))

      # update the model
      g = model.grad(s) # get the feature vector if this state
      err = target - Vs # update the error 
      model.w += ALPHA * err * g # update the model weigths
      
      # accumulate error
      n_steps += 1
      episode_err += err*err # square err

      # update state
      s = s2   # update the state
      Vs = Vs2 # update V(s)
    
    mse = episode_err / n_steps # compute the MSE of this epicode
    mse_per_episode.append(mse) # append that to the list


  ## Plot the data 
  plt.plot(mse_per_episode)
  plt.title("MSE per episode")
  plt.show()

  ## obtain predicted values
  V = {}                      # Initialize the V(s)
  states = grid.all_states()  # get all the states
  for s in states:            # Loop for each state
    if s in grid.actions:     # fs the state in the actions
      V[s] = model.predict(s) # predict the V(s) for this state
    else:      
      V[s] = 0 # terminal state or state we can't otherwise get to



  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(greedy_policy, grid)
