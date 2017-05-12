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
from keras.layers import Conv2D, Flatten, Dense

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
flags.DEFINE_string("mode", "test", "<train>, <test>, <random>")
flags.DEFINE_string("train_dir", "models/%s" % FLAGS.env_name, "directory path to save trained model")
flags.DEFINE_string("summary_dir", "summary/%s" % FLAGS.env_name, "directory path to save summary")

flags.DEFINE_float('gpu_memory_fraction', 0.2, 'gpu memory fraction.')

class Agent():
    def __init__(self, num_actions):
        # game
        self.num_actions = num_actions
        # frame index no. (mod interval_num でアクションを決定するときに使用)
        self.t = 0
        self.repeated_action = 0

        # summary
        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # replay memory
        self.replay_memory = deque()

        # train
        self.epsilon = FLAGS.initial_epsilon
        self.epsilon_step = (FLAGS.initial_epsilon - FLAGS.final_epsilon) / FLAGS.exploration_steps

        # create q-network
        self.s, self.q_values, q_network = self.build_model()
        q_network_weights = q_network.trainable_weights

        # crate target-network
        self.st, self.target_q_values, target_network = self.build_model()
        target_network_weights = target_network.trainable_weights

        # operation of updating target network
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # train_op
        self.a, self.y, self.loss, self.grad_update = self.build_training_op(q_network_weights)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, self.sess.graph)

        if not gfile.Exists(FLAGS.train_dir):
            gfile.MakeDirs(FLAGS.train_dir)

        self.sess.run(tf.global_variables_initializer())

        if FLAGS.load_network:
            model_dir = FLAGS.train_dir
            print("load_network: YES from %s" % (model_dir))
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Model restored.")
            else:
                print("No checkpoint file found")
        else:
            print("load_network: NO")

        # initialize target network q parameters = target parameters
        self.sess.run(self.update_target_network)


    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(FLAGS.env_name + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(FLAGS.env_name + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(FLAGS.env_name + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(FLAGS.env_name + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def build_model(self):
        '''
        build network model
        q and target network.
        :return: input data placeholder, inferenced q values, model
        '''
        # input datas placeholder
        s = tf.placeholder(tf.float32, [None, FLAGS.state_length, FLAGS.frame_width, FLAGS.frame_height])

        # model
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(FLAGS.state_length, FLAGS.frame_width, FLAGS.frame_height), data_format="channels_first"))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', data_format="channels_first"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', data_format="channels_first"))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        # inference of q-functions
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        # action
        a = tf.placeholder(tf.int64, [None])
        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)

        # target
        y = tf.placeholder(tf.float32, [None])

        # get q_value on action
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(FLAGS.lr, momentum=FLAGS.momentum, epsilon=FLAGS.min_grad)
        grad_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grad_update


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

    def get_action_at_test(self, state):
        action = self.repeated_action

        if self.t % FLAGS.action_interval == 0:
            if random.random() <= 0.05:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            self.repeated_action = action

        self.t += 1

        return action

    def get_action(self, state):
        action = self.repeated_action

        # mod action_intervalしかactionを更新しない
        if self.t % FLAGS.action_interval == 0:
            if self.epsilon >= random.random() or self.t < FLAGS.initial_replay_size:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            self.repeated_action = action

        # actionに依存して更新するepsilon値
        if self.epsilon > FLAGS.final_epsilon and self.t >= FLAGS.initial_replay_size:
            self.epsilon -= self.epsilon_step

        return action

    def task(self, state, action, reward, terminal, observation):
        '''
        次のアクションを状態
        :param state: 現在の状態
        :param action: 行動
        :param reward: 報酬
        :param terminal: ゲームの終端状態
        :param observation: 前処理後の画面
        :return: next_state
        '''
        next_state = np.append(state[1:, :, :], observation, axis=0)

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        # 報酬の固定、正は1、負は - 1、0 はそのままに変更
        reward = np.sign(reward)

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > FLAGS.num_replay_memory:
            self.replay_memory.popleft()

        # 初期に行うmemory保存用のフレーム数が過ぎてから学習
        if self.t >= FLAGS.initial_replay_size:
            # Train network
            if self.t % FLAGS.train_interval == 0:
                self.train_model()

            # Update target network
            if self.t % FLAGS.t_update_interval == 0:
                self.sess.run(self.update_target_network)
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % FLAGS.save_interval == 0:
                save_path = self.saver.save(self.sess, FLAGS.train_dir, global_step=(self.t))
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.duration += 1

        # ゲーム終端時
        if terminal:
            # Write summary（初期に行うmemory保存用のフレーム数が過ぎてから記録）
            if self.t >= FLAGS.initial_replay_size:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                         self.duration, self.total_loss / (float(self.duration) / float(FLAGS.train_interval))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            if self.t < FLAGS.initial_replay_size:
                mode = 'random(to save memory)'
            elif FLAGS.initial_replay_size <= self.t < FLAGS.initial_replay_size + FLAGS.exploration_steps:
                mode = 'explore(epsilon is big)'
            else:
                mode = 'train'
            print(
                'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                    self.episode + 1, self.t, self.duration, self.epsilon,
                    self.total_reward, self.total_q_max / float(self.duration),
                    self.total_loss / (float(self.duration) / float(FLAGS.train_interval)), mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_model(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Experience reply
        minibatch = random.sample(self.replay_memory, FLAGS.batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # 終了: 1、未終: 0
        terminal_batch = np.array(terminal_batch) + 0

        # target network から 推定のQ値 を取得
        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})

        # ground truth q-value（終了している場合はrewardのみになる）
        y_batch = reward_batch + (1 - terminal_batch) * FLAGS.gamma * np.max(target_q_values_batch, axis=1)

        # train network
        loss_val, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss_val


def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FLAGS.frame_width, FLAGS.frame_height)) * 255)
    return np.reshape(processed_observation, (1, FLAGS.frame_width, FLAGS.frame_height))


def train(env, agent):
    for _ in range(FLAGS.num_episodes):
        terminal = False
        observation = env.reset()

        # 最大30フレームスキップ（最初のフレームをランダムにするため）
        for _ in range(random.randint(1, FLAGS.no_op_steps)):
            last_observation = observation
            observation, _, _, _ = env.step(0)  # Do nothing
        state = Agent.get_initial_state(observation, last_observation)

        while not terminal:
            last_observation = observation

            action = agent.get_action(state)

            observation, reward, terminal, _ = env.step(action)
            processed_observation = preprocess(observation, last_observation)

            state = agent.task(state, action, reward, terminal, processed_observation)


def test(env, agent):
    for _ in range(FLAGS.test_step_number):
        terminal = False
        observation = env.reset()
        for _ in range(random.randint(1, FLAGS.no_op_steps)):
            last_observation = observation
            observation, _, _, _ = env.step(0)  # Do nothing
        state = agent.get_initial_state(observation, last_observation)
        while not terminal:
            last_observation = observation
            action = agent.get_action_at_test(state)
            observation, _, terminal, _ = env.step(action)
            env.render()
            processed_observation = preprocess(observation, last_observation)
            state = np.append(state[1:, :, :], processed_observation, axis=0)


def test_random(env, agent):
    for _ in range(FLAGS.test_step_number):
        terminal = False
        observation = env.reset()

        # 最大30フレームスキップ（最初のフレームをランダムにするため）
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


def main(_):
    env = gym.make(FLAGS.env_name)
    agent = Agent(num_actions=env.action_space.n)

    if FLAGS.mode == "train":
        train(env, agent)
    elif FLAGS.mode == "test":
        test(env, agent)
    elif FLAGS.mode == "random":
        test_random(env, agent)


if __name__ == "__main__":
    tf.app.run()


