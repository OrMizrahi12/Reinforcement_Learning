from __future__ import print_function, division
from builtins import range
import matplotlib.pyplot as plt
import numpy as np

"""
This file are show an example of how we can solve the
exploreations vs. exploitation, by the epsolin greedy. 

- Our goal is to chose the bigger probability: 0.75! 
"""


NUM_TRIALS = 10000 # number of trails
EPS = 0.1 # epcilon (10%)
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75] # initial bionomial probabilities


class BanditArm:

  # Initialize parameter
  def __init__(self, p):
    self.p = p           # the true win rate 
    self.p_estimate = 0. # the current estimate win rate
    self.N = 0.          # num samples collected so far
  
  # return 1/0 (true/false)
  def pull(self):
    # draw a 1 with probability p
    return np.random.random() < self.p
   
  # Function to update the estimate 
  def update(self, x):
    # update the number of samples (we just collect more one sumple)
    self.N += 1. 
    # update the current estimate win rate! 
    self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N

# 
def choose_random_argmax(a):
  idx = np.argwhere(np.amax(a) == a).flatten()
  return np.random.choice(idx)


# The experiment function:
def experiment():

  ### Initial parameters ###

  # initial the BanditArm objecy - with the respective probabilities 
  # for each probability, we initial the BanditArm object. 
  bandits = [BanditArm(p) for p in BANDIT_PROBABILITIES]
  rewards = np.zeros(NUM_TRIALS) # initial rewards array 
  num_times_explored = 0 # initial number of exploreations (it'll update)
  num_times_exploited = 0 # initial number of exploitation (it'll update)
  num_optimal = 0 # the number of times we chose the optimal
  optimal_j = np.argmax([b.p for b in bandits]) # optimal j - index corresponding to the band with the maximum true mean
  print("optimal j:", optimal_j)

  ### Implement tha epcolin greedy strategy ###
  
  # loop over the number of trails
  for i in range(NUM_TRIALS):

    # use epsilon-greedy to select the next bandit
    if np.random.random() < EPS: # less than epcolin ? find new once! 
      num_times_explored += 1 # update the exploration
      j = np.random.randint(len(bandits)) # chose new random act
   
    else: # bigger tha epcolin? choose the best.
      num_times_exploited += 1 # update the explodation
      # chose the bast probability (its argmax)! 
      # j = choose_random_argmax([b.p_estimate for b in bandits])
      j = np.argmax([b.p_estimate for b in bandits]) # more intuative 

    
    # if j (our chois) == to the optimal, we update the number of optimal
    if j == optimal_j:
      num_optimal += 1

    # pull the arm for the bandit with the largest sample
    x = bandits[j].pull()

    # update rewards log
    rewards[i] = x

    # update the distribution for the bandit whose arm we just pulled
    bandits[j].update(x)

    

  # print mean estimates for each bandit
  for b in bandits:
    print("mean estimate:", b.p_estimate)

  # print total reward
  print("total reward earned:", rewards.sum())
  print("overall win rate:", rewards.sum() / NUM_TRIALS)
  print("num_times_explored:", num_times_explored)
  print("num_times_exploited:", num_times_exploited)
  print("num times selected optimal bandit:", num_optimal)

  # plot the results
  cumulative_rewards = np.cumsum(rewards) 
  win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1) # cumelative sum of rewards 
  plt.plot(win_rates)
  plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
  plt.show()

if __name__ == "__main__":
  experiment()
