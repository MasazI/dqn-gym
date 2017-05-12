import os
import gym
import random
import numpy as np
from collections import deque

# tensorflow
import tensorflow as tf
from tensorflow.python.platform import gfile
flags = tf.app.flags
FLAGS = flags.FLAGS

# skimage
from skimage.color import rgb2gray
from skimage.transform import resize

# keras
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense

flags.DEFINE_string("env_name", "Breakout-v0", "environment name of gym")
flags.DEFINE_integer("frame_width", 84, "resiszed frame width")
flags.DEFINE_integer("frame_height", 84, "resized frame height")

flags.DEFINE_integer("num_episodes", 12000, "number of episodes")
flags.DEFINE_integer("state_length", 4, "number of most recent frames to produce the input to the network")
flags.DEFINE_float("gamma", 0.99, "discount factor")

flags.DEFINE_integer("exploration_steps", 1000000, "number of steps")
flags.DEFINE_float("initial_epsilon", 1.0, "initial value of epsilon-greedy's epsilon")
flags.DEFINE_float("final_epsilon", 0.1, "final value of epsilon-greedy's epsilon")

flags.DEFINE_integer("initial_replay_size", 20000, "number of steps to initialize replay memory")
flags.DEFINE_integer("num_replay_memory", 400000, "number of replay memory for training")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("t_update_interval", 10000, "target update interval")
flags.DEFINE_integer("action_interval", 4, "use only interval-th input")
flags.DEFINE_integer("train_interval", 4, "train interval-th actions between updates")
flags.DEFINE_float("lr", 0.00025, "learning rate")
flags.DEFINE_float("momentum", 0.01, "momentum for rmsprop")
flags.DEFINE_float("min_grad", 0.01, "min grad for rmsprop")
flags.DEFINE_integer("save_interval", 300000, "model save interval")
flags.DEFINE_integer("no_op_steps", 30, "maximum number of nothing actions")

flags.DEFINE_integer("test_step_number", 30, "number of episodes at test")

flags.DEFINE_boolean("load_network", False, "load network")
flags.DEFINE_string("mode", "random", "<train>, <test>, <random>")
flags.DEFINE_string("train_dir", "models/%s" % FLAGS.env_name, "directory path to save trained model")
flags.DEFINE_string("summary_dir", "summary/%s" % FLAGS.env_name, "directory path to save summary")


class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.t = 0
        self.repeated_action = 0

    @staticmethod
    def get_initial_state(observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FLAGS.frame_width, FLAGS.frame_height)) * 255)
        state = [processed_observation for _ in range(FLAGS.state_length)]
        return np.stack(state, axis=0)

    def get_action_random(self, state):
        action = self.repeated_action

        if self.t % FLAGS.action_interval == 0:
            action = random.randrange(self.num_actions)
            self.repeated_action = action

        self.t += 1

        return action


def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FLAGS.frame_width, FLAGS.frame_height)) * 255)
    return np.reshape(processed_observation, (1, FLAGS.frame_width, FLAGS.frame_height))


def train(env, agent):
    for _ in range(FLAGS.test_step_number):
        terminal = False
        observation = env.reset()

        for _ in range(random.randint(1, FLAGS.no_op_steps)):
            last_observation = observation
            observation, _, _, _ = env.step(0)  # Do nothing

        state = Agent.get_initial_state(observation, last_observation)

        while not terminal:
            last_observation = observation
            action = agent.get_action_random(state)
            observation, _, terminal, _ = env.step(action)
            env.render()
            processed_observation = preprocess(observation, last_observation)
            state = np.append(state[1:, :, :], processed_observation, axis=0)


def test(env, agent):
    pass


def test_random(env, agent):
    for _ in range(FLAGS.test_step_number):
        terminal = False
        observation = env.reset()

        for _ in range(random.randint(1, FLAGS.no_op_steps)):
            last_observation = observation
            observation, _, _, _ = env.step(0)  # Do nothing

        state = agent.get_initial_state(observation, last_observation)

        while not terminal:
            last_observation = observation
            action = agent.get_action_random(state)
            observation, _, terminal, _ = env.step(action)
            env.render()
            processed_observation = preprocess(observation, last_observation)
            state = np.append(state[1:, :, :], processed_observation, axis=0)


def main():
    env = gym.make(FLAGS.env_name)
    agent = Agent(num_actions=env.action_space.n)

    if FLAGS.mode == "train":
        train(env, agent)
    elif FLAGS.mode == "test":
        test(env, agent)
    elif FLAGS.mode == "random":
        test_random(env, agent)


if __name__ == "__main__":
    main()


