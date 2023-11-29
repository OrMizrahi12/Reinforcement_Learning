from __future__ import print_function, division
from builtins import range
import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
import tensorflow as tf

GAMMA = 0.99
ALPHA = 0.1


def epsilon_greedy(model, s, eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        values = model.predict_all_actions(s)
        return np.argmax(values)
    else:
        return model.env.action_space.sample()


def gather_samples(env, n_episodes=10000):
    samples = []
    for _ in range(n_episodes):
        s, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            a = env.action_space.sample()
            sa = np.concatenate((s, [a]))
            samples.append(sa)
            s, r, done, truncated, info = env.step(a)
    return samples


class Model(tf.keras.Model):
    def __init__(self, env, hidden_units=(32, 32), learning_rate=0.001):
        super(Model, self).__init__()
        self.env = env
        samples = gather_samples(env)
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dims = self.featurizer.n_components

        self.model = self.build_model(dims, hidden_units, learning_rate)

    def build_model(self, input_dim, hidden_units, learning_rate):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))

        for units in hidden_units:
            model.add(tf.keras.layers.Dense(units, activation='relu'))

        model.add(tf.keras.layers.Dense(1, activation=None))  # Linear activation for Q-value

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def predict(self, s, a):
        sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]
        x = np.expand_dims(x, axis=0)
        return self.model.predict(x)[0, 0]

    def predict_all_actions(self, s):
        return [self.predict(s, a) for a in range(self.env.action_space.n)]

    def grad(self, s, a):
        sa = np.concatenate((s, [a]))
        x = self.featurizer.transform([sa])[0]
        return x


def test_agent(model, env, n_episodes=20):
    reward_per_episode = np.zeros(n_episodes)
    for it in range(n_episodes):
        done = False
        truncated = False
        episode_reward = 0
        s, info = env.reset()
        while not (done or truncated):
            a = epsilon_greedy(model, s, eps=0)
            s, r, done, truncated, info = env.step(a)
            episode_reward += r
    return np.mean(reward_per_episode)


def watch_agent(model, env, eps):
    done = False
    truncated = False
    episode_reward = 0
    s, info = env.reset()
    while not (done or truncated):
        a = epsilon_greedy(model, s, eps=eps)
        s, r, done, truncated, info = env.step(a)
        episode_reward += r
    print("Episode reward:", episode_reward)


if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    model = Model(env)
    reward_per_episode = []

    watch_agent(model, env, eps=0)

    # repeat until convergence
    n_episodes = 1500
    for it in range(n_episodes):
        s, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            
            a = epsilon_greedy(model, s)
            s2, r, done, truncated, info = env.step(a)

            if done:
                target = r
            else:
                values = model.predict_all_actions(s2)
                target = r + GAMMA * np.max(values)

            g = model.grad(s, a)
            err = target - model.predict(s, a)
            model.model.train_on_batch(np.expand_dims(model.featurizer.transform([np.concatenate((s, [a]))])[0], axis=0), np.array([target]))

            episode_reward += r
            s = s2

        if (it + 1) % 50 == 0:
            print(f"Episode: {it + 1}, Reward: {episode_reward}")

        if it > 20 and np.mean(reward_per_episode[-20:]) == 200:
            print("Early exit")
            break

        reward_per_episode.append(episode_reward)

    test_reward = test_agent(model, env)
    print(f"Average test reward: {test_reward}")

    plt.plot(reward_per_episode)
    plt.title("Reward per episode")
    plt.show()

    # env = gym.make("CartPole-v1", render_mode="human")
    # watch_agent(model, env, eps=0)
