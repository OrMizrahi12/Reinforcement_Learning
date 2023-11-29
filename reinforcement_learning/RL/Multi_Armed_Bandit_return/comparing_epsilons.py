from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt

"""
- This file is all about comparing differences epsolins.
- our reward going to be Gaussian distribution.
  - Our goal is to reach the avarage value for each distribution.
"""

class BanditArm:
  
  # Initialize 
  def __init__(self, m):
    self.m = m          # true rate  
    self.m_estimate = 0 # its estimate (going to updating trugth the learning)
    self.N = 0          # number of sumples
   
  # function for get new value
  def pull(self):
    return np.random.randn() + self.m
  
  # Function to update the estimate avarage and the number of sumples.
  def update(self, x):
    self.N += 1
    self.m_estimate = (1 - 1.0/self.N)*self.m_estimate + 1.0/self.N*x


# Run the experiments:
def run_experiment(m1, m2, m3, eps, N):
  bandits = [BanditArm(m1), BanditArm(m2), BanditArm(m3)]

  # count number of suboptimal choices
  means = np.array([m1, m2, m3]) # the avarages
  true_best = np.argmax(means) # grab the true best
  count_suboptimal = 0

  data = np.empty(N)
  
  # Loop over number of trails
  for i in range(N):

    if np.random.random() < eps: # epsilon greedy
      j = np.random.choice(len(bandits))
      
    else: # choose the best
      j = np.argmax([b.m_estimate for b in bandits])
    
    x = bandits[j].pull() # get x (this is the true values (+-) with some noise  )
    bandits[j].update(x)  # update the estimate by x
    
    # If j is not the best true (that its actually the bigger avarage (3.5))
    if j != true_best:
      count_suboptimal += 1

    # for the plot
    data[i] = x
  
  cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

  # plot moving average ctr
  plt.plot(cumulative_average)
  plt.plot(np.ones(N)*m1)
  plt.plot(np.ones(N)*m2)
  plt.plot(np.ones(N)*m3)
  plt.xscale('log')
  plt.show()

  for b in bandits:
    print(b.m_estimate)

  print("percent suboptimal for epsilon = %s:" % eps, float(count_suboptimal) / N)

  return cumulative_average

if __name__ == '__main__':
  
  # set the The mean of each bandit
  m1, m2, m3 = 1.5, 2.5, 3.5
  c_1 = run_experiment(m1, m2, m3, 0.1, 100000)   # run experiment with epc=0.1
  c_05 = run_experiment(m1, m2, m3, 0.05, 100000) # run experiment with epc=0.5
  c_01 = run_experiment(m1, m2, m3, 0.01, 100000) # run experiment with epc=0.01

  # log scale plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(c_05, label='eps = 0.05')
  plt.plot(c_01, label='eps = 0.01')
  plt.legend()
  plt.xscale('log')
  plt.show()


  # linear plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(c_05, label='eps = 0.05')
  plt.plot(c_01, label='eps = 0.01')
  plt.legend()
  plt.show()

