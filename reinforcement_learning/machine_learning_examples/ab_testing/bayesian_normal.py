from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm





"""
Thompson Sampling With Gaussian Reward
=======================================

Intro
------

* Thompson Sampling (TS) is a Bayesian approach to the multi-armed bandit problem, 
  a classic problem in decision theory and reinforcement learning.

* In the multi-armed bandit problem, you are faced with a set of actions (arms), 
  each associated with an unknown reward distribution. 

* The goal is to learn which action is the most rewarding over time while balancing exploration 
  (trying new actions) and exploitation (choosing actions with high estimated rewards).

* Thompson Sampling with Gaussian Rewards is a specific version of Thompson Sampling tailored for situations 
  where the rewards associated with each arm are assumed to follow a Gaussian (normal) distribution.

  
A step-by-step explanation of how Thompson Sampling with Gaussian Rewards works
---------------------------------------------------------------------------------

1. Initialization:
  * Initialize prior distributions for the parameters of the reward distribution for each arm.
  * In the case of Gaussian rewards, this involves specifying prior distributions for the:
    - mean (μ)
    - variance (σ^2)
   
2. Sampling:
  * For each arm, sample a set of parameters (μ,σ^2) from its prior distribution.
    This is done independently for each arm.
  * This means that for each arm, 
    you create a candidate reward distribution based on the sampled parameters.

3. Selecting an Arm:
  * For each arm, generate a random sample from its candidate reward distribution. 
    This is effectively drawing a sample reward for each arm.

4. Choosing the Best Arm: 
  * Select the arm with the highest sampled reward. 
    This is the arm that the algorithm chooses to pull in the current round.

5. Observing the Outcome:
  * Pull the chosen arm and observe the actual reward.

6. Updating Priors:
  * Update the prior distributions for the chosen arm based on the observed reward. 
    This involves incorporating the observed data into the prior distribution, 
    often using Bayes' theorem.

7. Repeat:
  * Repeat the process for a specified number of rounds or until convergence.

"""




np.random.seed(1)

# number of trails
NUM_TRIALS = 2000 

# the TRUE mean of each arm:
# the goal is to predict the mean of each arm. 
# the algorithem does'nt know this values!!!! 
BANDIT_MEANS = [1, 2, 3]  
BANDIT_MEANS = [5, 10, 20] # try those  


class Bandit:
  def __init__(self, true_mean):
    """
    Bandit is an arm machine. 

    Parameters and Attributes
    -------------------------
    - `true_mean`: this is the true mean of the arm. 
    - `m`: 
      - Purpose: This attribute represents the predicted mean of the bandit's reward distribution.
      - Usage: It is used in the `sample()` method to generate random samples from the bandit's reward distribution.
      - Update: It is updated in the `update()` method based on the observed reward
    - `lambda_  (Precision or Precision Parameter)`: 
      - Purpose: This attribute is related to the precision (inverse of variance) of the predicted mean.
        A higher precision corresponds to lower uncertainty.
      - Usage: It is used in the `sample()` method to generate random samples from the bandit's reward distribution.
      - Update: It is updated in the update method. 
        The update increases the precision based on the bandit's existing precision and the precision of the new data.
    - `tau`: 
      - Purpose: Tau is a scaling factor used in the Bayesian updating process. 
        It influences how much the new data should impact the predicted mean.
      - Usage: It is used in the update method during the adjustment of the predicted mean.
      - Update: It remains constant and is used to scale the impact of new data. 
        A higher tau means that new data has a stronger influence on the predicted mean.
    - `N`: Number of Plays or Trials for each arm. 
    """
    self.true_mean = true_mean
    self.m = 0 # parameters for mu - prior is N(0,1)
    self.lambda_ = 1
    self.tau = 1 # 1 -> no effect. 
    self.N = 0


  def pull(self):
    """
    Drews a sample from the real normal distribution. (play a game in the arm!)
    We basically sample from a normal distribution with the true_mean. 
    """
    return np.random.randn() / np.sqrt(self.tau) + self.true_mean


  def sample(self):
    """
    Generate random sample from the bandit's reward distribution.
    """
    return np.random.randn() / np.sqrt(self.lambda_) + self.m


  def update(self, x):
    """
    Function for updating the `m` (predicted mean) and the `lambda_` (precision parameter)
    
    Parameters
    ----------
    - `x`: the reward from the arm 

    """
    # update the predicted mean:
    # tau, lambda_ (precision parameters) -> controls how much the bandit should trust its 
    #                                        existing predictions versus incorporating the new observed data.
    self.m = (self.tau * x + self.lambda_ * self.m) / (self.tau + self.lambda_)
    
    # updating the precision parameter associated with the predicted mean of the bandit's reward distribution.
    # Where the precision of its estimates increases, its indicating greater confidence in the predicted mean.
    # so naturally, wehere we run more sampling from a bandint, we get more accurate about the parameter ot the predicted mean.  
    self.lambda_ += self.tau
    self.N += 1


def plot(bandits, trial):
  """
  Function for plotting the distribution of a bandit. 
  """
  x = np.linspace(-3, 6, 200)
  for b in bandits:
    y = norm.pdf(x, b.m, np.sqrt(1. / b.lambda_))
    plt.plot(x, y, label=f"real mean: {b.true_mean:.4f}, num plays: {b.N}")
  plt.title(f"Bandit distributions after {trial} trials")
  plt.legend()
  plt.show()


def run_experiment():
  bandits = [Bandit(m) for m in BANDIT_MEANS]
  
  # milestone point for plot the distributions of the bandits.
  sample_points = [5,10,20,50,100,200,500,1000,1500,1999]

  rewards = np.empty(NUM_TRIALS)
  for i in range(NUM_TRIALS):
    # Thompson sampling (take the arm with the higher sample. (the best arm.))
    j = np.argmax([b.sample() for b in bandits])

    # plot the posteriors 
    if i in sample_points:
      plot(bandits, i)

    # pull the arm for the bandit with the largest sample
    x = bandits[j].pull()

    # update the distribution for the bandit whose arm we just pulled
    bandits[j].update(x)

    # update rewards
    rewards[i] = x

  cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

  # plot moving average ctr
  plt.plot(cumulative_average)
  for m in BANDIT_MEANS:
    plt.plot(np.ones(NUM_TRIALS)*m)
  plt.show()

  return cumulative_average

if __name__ == '__main__':
  run_experiment()


