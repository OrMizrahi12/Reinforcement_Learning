import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler



"""
1. Utilities Functions:
========================
"""

# Let's use AAPL (Apple), MSI (Motorola), SBUX (Starbucks)
def get_data():
  """
  Function get the stock data
  
  return list with shape of (n rows, 3 column).
  - each row: is a day
  - each column: is a stock
  """
  df = pd.read_csv('aapl_msi_sbux.csv') # read the file
  return df.values  # return its numpy array



def get_scaler(env):
  """
  return scikit-learn scaler object to scale the states
  
  - Note: you could also populate the replay buffer here
  """
  
  states = [] # initialize list of states
  for _ in range(env.n_step): # loop of all the steps in the env
    action = np.random.choice(env.action_space)  # choise a random action 
    state, reward, done, info = env.step(action) # perform the action
    states.append(state) #  append the state
    if done:
      break
  
  scaler = StandardScaler() # create scaler instance
  scaler.fit(states)        # scale the data (the states)
  return scaler             # return the scalered states. 


def maybe_make_dir(directory):
  """
  maybe make dir
  ---------------
  - if the directory do not exsist, create one. 
  """
  if not os.path.exists(directory):
    os.makedirs(directory)


"""
2. The linear model class:
========================
"""
import tensorflow as tf

class ANNModel:
  
    def __init__(self, input_dim, n_action, hidden_units=64, learning_rate=0.01, momentum=0.9):
        # Define the model architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(n_action)
        ])

        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss='mse')

        # Initialize losses list
        self.losses = []

    def predict(self, X):
        # Make predictions using the model
        return self.model.predict(X)

    def sgd(self, X, Y, batch_size=32, epochs=1):
        # Train the model using Stochastic Gradient Descent
        history = self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=0)
        
        # Append the loss to the losses list
        self.losses.extend(history.history['loss'])

    def load_weights(self, filepath):
        # Load weights from a file
        self.model.load_weights(filepath)

    def save_weights(self, filepath):
        # Save weights to a file
        self.model.save_weights(filepath)


"""
3. The Trading Enviroment
==========================
"""


class MultiStockEnv:
  """
  A 3-stock trading environment.
  ==============================

### State: vector of size `7` (n_stock * 2 + 1)
    1. shares of stock 1 owned
    2. shares of stock 2 owned
    3. shares of stock 3 owned
    4. price of stock 1 (using daily close price)
    5. price of stock 2
    6. price of stock 3
    7. cash owned (can be used to purchase more stocks)

### Action: categorical variable with 27 (3^3) possibilities
  for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy

### Parameters
    - `data` the financial data (typically stock price history)
    - `initial_investment`: the money that you want to trade (deafult 20,000$)
  """
  def __init__(self, data, initial_investment=20000):
    # assigtn the data
    self.stock_price_history = data 
    # extract the number of steps and number of stock from the price hostory shape
    # n_step  -> the rows, (typically the days)
    # n_stock -> the columns (number of instriments)
    self.n_step, self.n_stock = self.stock_price_history.shape # 

    # instance attributes
    self.initial_investment = initial_investment
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None
    
    # define action space.
    self.action_space = np.arange(3**self.n_stock)

    # action permutations: all the possible combinations of the actions
    # 0 = sell, 1 = hold, 2 = buy
    # returns a nested list with elements like:
    # [0,0,0]
    # [0,0,1]
    # [0,0,2]
    # [0,1,0]
    # [0,1,1]
    # ...
    self.action_list = list(
      map(list, itertools.product([0, 1, 2], repeat=self.n_stock))
    )

    # calculate size of state
    # n_stock * 2 = number of 3 stock + their price = 6 
    # 1 = cash owned
    # so the size of state is 7. 
    self.state_dim = self.n_stock * 2 + 1
    
    # reset the env
    self.reset()


  def reset(self):
    """
    reset the state
    """
    self.cur_step = 0 # set current step to 0. this means that we point to the first day of stock prices in our dataset.
    self.stock_owned = np.zeros(self.n_stock) # stock owned to an array of all zeros. we dont hold any stock. we have 0 shares from each stock. 
    self.stock_price = self.stock_price_history[self.cur_step] # the stock price that correspomsed to the step. in this case - the stock price of the first day. 
    self.cash_in_hand = self.initial_investment # assign initial investment
    return self._get_obs() # return the state vector which is done by using the function to get ups.

  
  def step(self, action):
    """
    performs an action in the environment and returns the `next state` and `reward`.
    """
    # Check if the action that we want to perform exsist in the action space.
    # if not we assert an err, beacuse this action doesnot exsist. 
    assert action in self.action_space
    
    # get current value of our portfolio before performing the action
    prev_val = self._get_val()

    # update price, i.e. go to the next day
    self.cur_step += 1 # increment the current step (look at the next day) 
    self.stock_price = self.stock_price_history[self.cur_step] # look at the price that curresponsed to this step (e.g: step 2, show the price of day 2)

    # perform the trade! 
    self._trade(action)

    # get the new value after taking the action
    # this is the current value of the portfolio, after taking the action (the trade).
    cur_val = self._get_val()

    # reward is the increase in porfolio value
    # the reward is basically the different between the: 
    # current portfolio value and the previuse portfolio value.
    reward = cur_val - prev_val

    # done if we have run out of data
    done = self.cur_step == self.n_step - 1

    # store the current value of the portfolio here
    info = {'cur_val': cur_val}

    # conform to the Gym API
    # return the next state vector, reward, if its done, and info.
    # (just like the openai gym API)
    return self._get_obs(), reward, done, info


  def _get_obs(self):
    """
    Return the state as a vector (1D numpy array).
    - Basically, we store the state attribute in a vector like this example:
      - `array([1,2,1,200$,190$,100$,200000$])`
        - `1,2,1` -> the stocks we owned (1 AAPL,...,...)
        - `200$,190$,100$` the price of each instrument
        - `200000$`: the cash we have
    """
    obs = np.empty(self.state_dim) # initialize vector for store the state attributes 
    obs[:self.n_stock] = self.stock_owned # store the number of stocks
    obs[self.n_stock:2*self.n_stock] = self.stock_price # store the price of each instrument
    obs[-1] = self.cash_in_hand # finally, store the cash.
    
    # return the state vector 
    return obs
    


  def _get_val(self):
    """
    return the current value of our portfolio. 
    """
    # number of stock we owned * price of each stock + the cash in hand.  
    return self.stock_owned.dot(self.stock_price) + self.cash_in_hand


  def _trade(self, action):
    """
    Function for execute a trade

    Parameters
    ----------
    - `action`: the action we want to perform (in this case, is list)
      - the action, going to be an integer index.
      - e.g. action_space[index] = [2,1,0] means: buy first stock, hold second stock, sell third stock
    """

    # index the action we want to perform
    # 0 = sell, 1 = hold, 2 = buy
    # e.g. [2,1,0] means: buy first stock, hold second stock, sell third stock
    action_vec = self.action_list[action]

    # determine which stocks to buy or sell
    sell_index = [] # stores index of stocks we want to sell
    buy_index = []  # stores index of stocks we want to buy

    for i, a in enumerate(action_vec):
      if a == 0: # if action = 0 (sell)
        sell_index.append(i) # push the index that represent the stock to the sell index list
      elif a == 2: # if action == 2 (buy)
        buy_index.append(i)  # push the index that represent the stock to the buy index list

    # sell any stocks we want to sell
    if sell_index: # if we want to sell somting...
      # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
      for i in sell_index:
        # update the cash:
        # stock_price[i] -> the price of the stock (each stock have its own index)
        # stock_owned[i] -> the number of this specific stock. 
        self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        # now we dont hold this instrument any more. we sell him.
        self.stock_owned[i] = 0
    
    # then buy any stocks we want to buy
    if buy_index: # if we wabt to but somting...
      # NOTE: when buying, we will loop through each stock we want to buy,
      #       and buy one share at a time until we run out of cash
      can_buy = True
      while can_buy: # buy as we can
        for i in buy_index: # loop over the indexes that represent the stocks that we want to buy
          if self.cash_in_hand > self.stock_price[i]: # if the cash bigger tham the stock price
            self.stock_owned[i] += 1                  # buy one share
            self.cash_in_hand -= self.stock_price[i]  # update the cash
          else: # if we dont have a money... 
            can_buy = False


"""
The artificial intelligence agent
=========================
"""

class DQNAgent:

    def __init__(self, state_size, action_size, hidden_units=64, learning_rate=0.01, momentum=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Initialize the ANNModel
        self.model = ANNModel(state_size, action_size, hidden_units=hidden_units, learning_rate=learning_rate)

    def act(self, state):
        # Exploration
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        # Exploitation
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Returns the best action

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target

        # Train the ANNModel using Stochastic Gradient Descent
        self.model.sgd(state, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        # Load weights for the ANNModel
        self.model.load_weights(name)

    def save(self, name):
        # Save weights for the ANNModel
        self.model.save_weights(name)

"""
========================================================================
"""
def play_one_episode(agent, env, is_train):
  """
  Play one epicode.

  Parameters
  -----------
  - `agent`: the AI agent.
  - `env`: the enviroment where the agent'll interact
  - `is_train`: if we want to train the agent. 
  """
  # note: after transforming states are already 1xD
  state = env.reset()               # reset the env, get the start state
  state = scaler.transform([state]) # transform the state
  done = False
  limit = 0
  # While its not done:
  while not done: 
    limit = limit + 1
    action = agent.act(state) # given a state, give me an action
    next_state, reward, done, info = env.step(action) # perform the action
    next_state = scaler.transform([next_state])       # transform the next state
    if is_train == 'train': # if we want also to train the model...
      agent.train(state, action, reward, next_state, done)
    # update the state to be the next state 
    state = next_state
    
    # .... 

  # after the epicode, we return the current value of the portfolio. 
  return info['cur_val']





"""
Running all the plane
=====================

"""

if __name__ == '__main__':

  # configuration variables:
  models_folder = 'ann_rl_trader_models'   # where we'll save our models.
  rewards_folder = 'ann_rl_trader_rewards' # where we'll store our rewards.
  num_episodes = 1 # number of epicode
  batch_size = 32     # batch size (for sampling from the replay memory)
  initial_investment = 20000 # initial investment of the portfolio
  
  """
  ======================================
  """
  # we create an argument past the object so that we can run the script with command line arguments.
  
  # parser = argparse.ArgumentParser()
  # parser.add_argument('-m', '--mode', type=str, required=True,
  #                     help='either "train" or "test"')
  # args = parser.parse_args()

  from types import SimpleNamespace

  args = SimpleNamespace()
  # toogle that..
  # args.mode = "test"
  args.mode = "train"
  """
  ======================================
  """
  # Create the directories (models/rewards)
  maybe_make_dir(models_folder)
  maybe_make_dir(rewards_folder)
  
  # get the data
  data = get_data()
  n_timesteps, n_stocks = data.shape
  
  # split the data into train and test (50%/50%)
  n_train = n_timesteps // 2
  train_data = data[:n_train]
  test_data = data[n_train:]
  
  # Create the env with the training data
  env = MultiStockEnv(train_data, initial_investment)
  state_size = env.state_dim # extract the state size (how many attrivutes we have in a state?)
  action_size = len(env.action_space) # extract the action size
  agent = DQNAgent(state_size, action_size) # now, create an agent, define the input/output shape. (input - state size, output - action size!)
  scaler = get_scaler(env) # scale the enviroment

  # store the final value of the portfolio (end of episode)
  portfolio_value = []
  
  """
  Very importent IF statment!
  ---------------------------
  """
  # if we're in test mode, we want to use the scalar that we had 
  # during training so that the neural network corresponds with the 
  # same scalar and we don't accidentally use a different scalar.
  
  
  if args.mode == 'test':
    # then load the previous scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)

    # create the env with test data
    env = MultiStockEnv(test_data, initial_investment)

    # make sure epsilon is not 1!
    # no need to run multiple episodes if epsilon = 0, it's deterministic
    agent.epsilon = 0.01

    # load trained weights
    agent.load(f'{models_folder}/linear.npz')

  # play the game num_episodes times
  for e in range(num_episodes):
    
    t0 = datetime.now() # for know the time for each loop iteration..
    val = play_one_episode(agent, env, args.mode) # play one epicode
    dt = datetime.now() - t0 # measure the time different (how much take to play one..)
    
    # inform how many time take make one epicode
    print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
    
    # append episode end portfolio value (the portfolio value after one epicode)
    portfolio_value.append(val) 

  # save the weights when we are done
  if args.mode == 'train':
    # save the DQN
    agent.save(f'{models_folder}/linear.npz')

    # save the scaler also
    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)

    # plot losses
    plt.plot(agent.model.losses)
    plt.show()


  # save portfolio value for each episode
  np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)



# """
# What to run for play this majic?
# >>> python linear_rl_trader.py -m train && python plot_rl_rewards.py -m train
# """

