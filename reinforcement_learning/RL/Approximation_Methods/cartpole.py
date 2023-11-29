from __future__ import print_function, division
from builtins import range
import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler


"""
Cart Pole 
=========

we are going to implement cart pule, using the enviremonet of openAI gym,
and Approximation control RL algoriten. 
"""

GAMMA = 0.99
ALPHA = 0.1


def epsilon_greedy(model, s, eps=0.1):
  """
  Implement the epsilon greedy, for balance eplore & exploite 
  """
  p = np.random.random()
  if p < (1 - eps):
    values = model.predict_all_actions(s)
    return np.argmax(values)
  else:
    return model.env.action_space.sample()


def gather_samples(env, n_episodes=10000):
  """
  For cellect samples.

  Return -> [[state,action], ..., [state,action]]
  """
  samples = []
  for _ in range(n_episodes): # loop over `n` epicode
    s, info = env.reset()     # reset the env & take the start state 
    done = False 
    truncated = False
    while not (done or truncated):  # while the game not finish
      a = env.action_space.sample() # take the action, given the state
      sa = np.concatenate((s, [a])) # concat the state and action 
      samples.append(sa)            # append the sample

      s, r, done, truncated, info = env.step(a) # take step in the env
  return samples


class Model:
  def __init__(self, env):
    """
    A Reinforcement Learning model using function approximation for 
    solving a control problem in a given environment.
    """
    ## fit the featurizer to data
    self.env = env                      # set env
    samples = gather_samples(env)       # get and set the samples
    self.featurizer = RBFSampler()      # create instance of the featurizer
    self.featurizer.fit(samples)        # for the sample for create features (by the featurizer)
    dims = self.featurizer.n_components # number of features dimensions 

    self.w = np.zeros(dims) # initialize linear model weights

  def predict(self, s, a):
    """
    Prediction for a State-Action Pair
    """
    sa = np.concatenate((s, [a]))          # concatinate the [state,action]
    x = self.featurizer.transform([sa])[0] # create features from [state,action]
    return x @ self.w                      # return the dot of the features and the model weigths 
   
  def predict_all_actions(self, s):
    """
    Predict the State-Action pair all the action in a given state

    returns a list containing the predicted values for all possible actions in the given state.
    """
    # predict the value for spesific state with all the actions. 
    # (0,1) @ 'U' , (0,1) @ 'D', (0,1) @ 'L' , ...
    return [self.predict(s, a) for a in range(self.env.action_space.n)]

  def grad(self, s, a):
    """
    For create features based on state and action. 
    """
    sa = np.concatenate((s, [a]))
    x = self.featurizer.transform([sa])[0]
    return x



def test_agent(model, env, n_episodes=20):
  """
  Function for test the agent
  
  After the agent trained, we test it. 

  - Note: we set the epsolin to 0 (`epsilon_greedy(model, s, eps=0)`),
    beacuse its a test mode. so we want exploite only. 
  """
  reward_per_episode = np.zeros(n_episodes) # list for store the rewards per epicode
  for it in range(n_episodes): # For each epicode
    done = False       # stop game condition 1
    truncated = False  # stop game condition 2
    episode_reward = 0 # initial reward for this epicode 
    s, info = env.reset()          # get the start state
    
    while not (done or truncated): # while the game is not over... 
      a = epsilon_greedy(model, s, eps=0)       # based on the model policy and the state, get an action. 
      s, r, done, truncated, info = env.step(a) # perform the action, get back: state, reward, if the game done 
      episode_reward += r                       # update the reward of this epicode
    
    reward_per_episode[it] = episode_reward     # inseret the final reward of the epicode to the list 
  
  return np.mean(reward_per_episode)            # return the mean reward of the epicodes. 


def watch_agent(model, env, eps):
  """
  Function that playing one epicode for train the RL model. 
  
  """
  done = False          # stop game condition 1 
  truncated = False     # stop game condition 2
  episode_reward = 0    # reward for this epicode
  s, info = env.reset() # rest env & get the start state
  while not (done or truncated): # while the game is not over
    a = epsilon_greedy(model, s, eps=eps)     # given the model and state, get an action
    s, r, done, truncated, info = env.step(a) # perform the action
    episode_reward += r                       # update the reward 
  
  print("Episode reward:", episode_reward)    # print final (the total) reward of this epicode.


"""
Implement the approximate control RL algorithem
-----------------------------------------------
"""
if __name__ == '__main__':
  
  env = gym.make("CartPole-v1", render_mode="rgb_array") # instantiate environment

  model = Model(env)      # Create the model
  reward_per_episode = [] # initialize list for store the rewars pwe epicode

  watch_agent(model, env, eps=0) # watch untrained agent (train the agent one epicode)

  # repeat until convergence
  n_episodes = 1500
  for it in range(n_episodes): # loop over the epicodes
    s, info = env.reset() # reset the env & get the start state
    episode_reward = 0    # reset epicode reward 
    done = False          # reset stop game condition 1
    truncated = False     # reset stop game condition 2
    
    while not (done or truncated): # while the game not over... 
      a = epsilon_greedy(model, s) # given the model and state, return and action
      s2, r, done, truncated, info = env.step(a) # perform the action, get back: state, reward, stop condition. 

      # get (update) the target
      if done:
        target = r
      else:
        values = model.predict_all_actions(s2) # predict the values of this state with all possible actions [(0,1), 'U', ... , (0,1), 'D']
        target = r + GAMMA * np.max(values)    # update the target with the value that yield from the best action in this state  

      # update the model
      g = model.grad(s, a)               # get features based on [state,action]
      err = target - model.predict(s, a) # update the err (the different between the actual to the prediction value.)
      model.w += ALPHA * err * g         # update the model weigths 
      
      # accumulate reward
      episode_reward += r

      # update state
      s = s2

    if (it + 1) % 50 == 0:
      print(f"Episode: {it + 1}, Reward: {episode_reward}")

    # early exit
    if it > 20 and np.mean(reward_per_episode[-20:]) == 200:
      print("Early exit")
      break
    
    reward_per_episode.append(episode_reward) # append the reward epicode to the list

  
  test_reward = test_agent(model, env) # test trained agent
  print(f"Average test reward: {test_reward}")

  plt.plot(reward_per_episode)
  plt.title("Reward per episode")
  plt.show()

  # watch trained agent
  env = gym.make("CartPole-v1", render_mode="human")
  watch_agent(model, env, eps=0)
