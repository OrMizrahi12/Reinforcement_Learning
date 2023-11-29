import matplotlib.pyplot as plt
import numpy as np
import argparse

"""
For plotting the rewards we saved. 
its print the avarage reward, the min, and the max. 
"""

# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--mode', type=str, required=True,
#                     help='either "train" or "test"')
# args = parser.parse_args()

from types import SimpleNamespace

args = SimpleNamespace()
# toogle that..
# args.mode = "test"
args.mode = "train"

a = np.load(f'ann_rl_trader_rewards/{args.mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

if args.mode == 'train':
  # show the training progress
  plt.plot(a)
else:
  # test - show a histogram of rewards
  plt.hist(a, bins=20)

plt.title(args.mode)
plt.show()