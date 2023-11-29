from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt


"""
UCB1 (Upper Confidence Bound 1) 
===============================

Intro
-----

* The UCB1 (Upper Confidence Bound 1) algorithm is a popular algorithm used in the context of multi-armed bandit problems, 
  which are a class of reinforcement learning problems. In a multi-armed bandit problem, 
  an agent is faced with a set of slot machines (arms), each with an unknown probability distribution of yielding rewards. 
  The goal is for the agent to maximize its total reward over a series of trials.
    - In other words, 
      the goal of the agent is to find the arm (slot machine) that has the highest probability of yielding a reward.

* The UCB1 algorithm is designed to balance exploration (trying different arms to learn their characteristics)
  and exploitation (choosing the arm that is believed to be the best based on current knowledge)
  to achieve optimal performance over time.


high-level explanation 
----------------------
* Initialization:   
  - Initialize an array to store the number of times each arm has been pulled (pull_count)
    and the sum of rewards obtained from each arm (reward_sum).

* Exploration-Exploitation Trade-off:
  - In each round, calculate the UCB value for each arm using the following formula:
    - UCB[i] = X[i] + sqrt( (2*log(t)) / n[i]  )
      - Where: 
        - UCB[i]: is the upper confidence bound for arm[i]
        - X[i]:   is the avarage reward from arm[i] ( reward_sum / pull_count) 
        - n[i]:   is the number of times that arm[i] has bean pulled
        - t:      is the total number of rounds

* Arm Selection (choose the best arm after we've run the algorithem):
  - Choose the arm with the highest UCB value.
  - This balances the desire to choose arms with potentially high rewards (exploitation) 
    and the need to explore arms to refine their reward estimates.

Conclusion
-----------
1. The UCB1 algorithm ensures that arms that have not been explored much or have uncertain 
  reward estimates are given higher priority, promoting exploration. 

2. As the number of rounds increases, the algorithm tends to converge towards 
   exploiting the arm with the highest estimated reward.
"""


NUM_TRIALS = 100000 # number of trails
EPS = 0.1 # epcilon  
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75] # probabilities (of curse, we want choose the best armsproba - 0.75)


class Bandit:

  # Initialize
  def __init__(self, p):  
    self.p = p           # the true win rate
    self.p_estimate = 0. # the estimate (will update over the time)
    self.N = 0.          # num samples collected so far
  
  # Pulling function: based on the probability of the arm, return loss/win!
  def pull(self):
    # draw a 1 with probability p
    return np.random.random() < self.p 
   
  # Updating the estimate  
  def update(self, x):
    self.N += 1.
    self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N

# UCG algorithem
def ucb(mean, n, nj):
  return mean + np.sqrt(2*np.log(n) / nj)


# Run the experiment:
def run_experiment():
  bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
  rewards = np.empty(NUM_TRIALS)
  total_plays = 0

  # initialization: play each bandit once
  for j in range(len(bandits)): # for each bandit (arms) 
    x = bandits[j].pull() # 1. pull (play a game in the arm (slot machine) ) 
    total_plays += 1      # 2. update the total plays
    bandits[j].update(x)  # 3. update the estimate probabiliry of this arm (slot machine). 
                          #    after each game in that arm, the probability change,
                          #    and we need to update the most recent probability of this arms. 
  

  # now, run NUM_TRIALS
  for i in range(NUM_TRIALS):
    j = np.argmax([ucb(b.p_estimate, total_plays, b.N) for b in bandits]) # 1. take the argmax of the UCB output.
    x = bandits[j].pull() # 2. pull (take a game with that arm (slot machine))
    total_plays += 1      # 3. update total plays
    bandits[j].update(x)  # 4. update the current estimate probability

    rewards[i] = x # for the plot
  
  # Compute the cumelatuve avarage
  cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

  # plot moving average ctr
  plt.plot(cumulative_average)
  plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
  plt.xscale('log')
  plt.show()

  # plot moving average ctr linear
  plt.plot(cumulative_average)
  plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
  plt.show()

  for b in bandits:
    print(b.p_estimate)

  print("total reward earned:", rewards.sum())
  print("overall win rate:", rewards.sum() / NUM_TRIALS)
  print("num times selected each bandit:", [b.N for b in bandits])

  return cumulative_average

if __name__ == '__main__':
  run_experiment()

