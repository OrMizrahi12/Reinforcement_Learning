from __future__ import print_function, division
from builtins import range
import matplotlib.pyplot as plt
import numpy as np

# The beta module of only for drawing the PDF of the beta distribution for each bandit.
from scipy.stats import beta


"""
Bayesian bandit
================

Intro
------

* Bayesian bandits are a class of algorithms used in sequential decision-making problems,
  particularly in scenarios where there's a need to balance exploration and exploitation.

* In the Bayesian context, the algorithm maintains a probability distribution over the 
  unknown parameters (such as the mean or expected reward) of each arm.

* As new information is obtained through interactions (pulling arms and observing rewards), 
  the probability distributions are updated, allowing the algorithm to make more informed decisions.
  

Math
-----

* θ[i]:   the unknown parameter (e.g., mean reward) for arm[i]
* X[i,t]: the observed reward from pulling arm[i] at time t. 
* D[i,t]: he history of observed rewards for arm[i] up to time t. 

 The probability distribution over the unknown parameter θ[i],
 denoted as: `P( θ[i] | D[i,t] )`, representing our belief about θ[i], given the observed data.
 
 Bayesian bandits update these probability distributions using Bayes' theorem after each round.
 The update can be expressed as:

 * P( θ[i] | D[i,t+1] ) = ( P( X[i,t+1] |  θ[i] D[i,t] ) * P( θ[i] | D[i,t] ) ) 
                        _________________________________________________
                                    P( X[i,t+1] |  D[i,t] ) 
  Where:
    - P( θ[i] | D[i,t] ):           is the posterior distribution after observing a new data point.
    - P( X[i,t+1] |  θ[i] D[i,t] ): is the likelihood of observing the new data point given the parameter.
    - P( θ[i] | D[i,t+1] ):         is the prior distribution representing our belief before observing the new data poin
    - P( X[i,t+1] |  D[i,t] ):      is the marginal likelihood (the probability of observing the new data point under all possible parameter values).

  
  After updating the distributions, the algorithm decides which arm to pull next 
  based on a criterion that balances exploration (choosing arms with uncertain parameters)
  and exploitation (choosing arms with high estimated rewards).


The main idea of this algorithem is:
- Based on exploration and exploitation, find the true distribution of each machine. 

E.g: if the true mean of a machine is 0.75, we want that the algorithem'll find it by itself, 
     by exploration and exploitation. 

So, each exploration and exploitation, the algorithem'll get the "true picture" if each distribution.  

"""



np.random.seed(2)

# number of trails
NUM_TRIALS = 2000  

# The bandit probabilies
# this our 3 slot machines: each machine have its probability of win.
# e.g.: the first slot machine have probability of 0.2 to win. 
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75] 



class Bandit:
  def __init__(self, p):
    """
    Create a bandint (slot machine). 
    ---------------------------------

    - `p`: the true probability of the machine
    - `a`: used to keep track of the number of successes (or positive outcomes) associated with a particular bandit. 
    - `b`: used to keep track of the number of failures (or negative outcomes) associated with a particular bandit.
      - Together, these parameters are used to model a Beta distribution for each bandit. 
        The Beta distribution is updated as more data is observed (through the update method),
        and it is sampled from in the sample method.
    - `N`: for information only (how mant time we play with the bandint?)
    """
    self.p = p 
    self.a = 1 
    self.b = 1 
    self.N = 0  

  def pull(self):
    """
    play a game on the machine.

    the more probability of a bandit to win (e.g 0.75), the more probability to get 1/true.
    """
    return np.random.random() < self.p


  def sample(self):
    """
    generates a random sample from a beta distribution with parameters a,b.
    Each bandit (slot machine) have its own distribution of sucess and filed. 

    - self.a: The number of successes or positive outcomes for a particular arm. 
    - self.b: The number of failures or negative outcomes for the same arm.
    """  
    return np.random.beta(self.a, self.b)
  
  
  def update(self, x):
    """
    Function for update the distributions parameters: `a` and `b` for each arm (slot machine).
    this is update the dostribution of success and filed. 
    
    So `a`,`b` is represent the number of seccess/filad. its should look like this: a=1230 success.., b=100 fialds..
      so its mean that this machine have 1230 wins and 100 loss, at this point.  
    
    - `self`: is the Bendit (slot machine)
    """
    self.a += x     # update the probability of success.
    self.b += 1 - x # update the probability of field.
    self.N += 1     # update the number of play game (for info only.)

# Plot
def plot(bandits, trial):
  x = np.linspace(0, 1, 200) # create linear space

  for b in bandits: # for each bandint (arm / slot machinne)
    y = beta.pdf(x, b.a, b.b) # create distribution with its own parameter (a: success, b: faild parameter) 
    plt.plot(x, # the linear space
             y, # the distribution
             label=f"real p: {b.p:.4f}, win rate = {b.a - 1}/{b.N}")

  plt.title(f"Bandit distributions after {trial} trials")
  plt.legend()
  plt.show()


def experiment():

  # initialize the bandits
  bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
  
  # mile stone points that we want to plot
  # plot after 200 sampling..
  sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
  
  rewards = np.zeros(NUM_TRIALS)
  
  # loop over trails
  for i in range(NUM_TRIALS):

    # Thompson sampling:
    # take the argmax of the bandit (the slot machine) that return the higher probability to win 
    # Basically it says give us the bandit that yields the largest sample from its current beta posterior.
    j = np.argmax([b.sample() for b in bandits])

    # plot the posteriors
    if i in sample_points:
      plot(bandits, i)

    # pull the arm for the bandit with the largest sample
    x = bandits[j].pull()

    # update rewards
    rewards[i] = x

    # update the distribution for the bandit whose arm we just pulled
    bandits[j].update(x)

  # print total reward
  print("total reward earned:", rewards.sum())
  print("overall win rate:", rewards.sum() / NUM_TRIALS)
  print("num times selected each bandit:", [b.N for b in bandits])


if __name__ == "__main__":
  experiment()

