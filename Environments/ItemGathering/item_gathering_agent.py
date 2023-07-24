import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Dense
from collections import deque  # Used for replay buffer and reward tracking
from Environments.ItemGathering.item_gathering_env import ItemGathering
from datetime import datetime  # Used for timing script
import time
from scipy.special import softmax

# 5.3 hours
SEED = 42
DEBUG = False
TEMPERATURE = 2
BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 6_000

GAMMA = 0.99

TRAINING_EPISODES = 200_000
EXPLORATION_RESTARTS = 0

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1 / 50000

COPY_TO_TARGET_EVERY = 1000  # Steps
START_TRAINING_AFTER = 50  # Episodes
FRAME_STACK_SIZE = 3
NUM_WEIGHTS = 6


class ReplayMemory(deque):
    """
    Inherits from the 'deque' class to add a method called 'sample' for
    sampling batches from the deque.
    """

    def sample(self, batch_size):
        """
        Sample a minibatch from the replay buffer.
        """
        # Random sample of indices
        indices = np.random.randint(len(self), size=batch_size)
        # Filter the batch from the deque
        batch = [self[index] for index in indices]
        # Unpach and create numpy arrays for each element type in the batch
        states, actions, rewards, next_states, dones, weightss = [
            np.array([experience[field_index] for experience in batch]) for field_index in range(6)]
        return states, actions, rewards, next_states, dones, weightss


class PreferenceSpace:
    def sample(self):
        # Each preference weight is randomly sampled between -20 and 20 in steps of 5
        p_step = 1  # Time Penalty
        p_wall = 5  # Wall
        p_green = random.choice([x for x in range(0, 20) if x % 5 == 0])  # Green
        p_red = random.choice([x for x in range(0, 20) if x % 5 == 0])  # Red
        p_yellow = random.choice([x for x in range(0, 20) if x % 5 == 0])  # Yellow
        p_other_red = random.choice([x for x in range(-20, 20) if x % 5 == 0])  # Other Red
        preference = np.array([p_step, p_wall, p_green, p_red, p_yellow, p_other_red], dtype=np.float32)
        return preference


class DQNAgent:

    def __init__(self, env, model_path):
        self.env = env
        self.actions = [i for i in range(self.env.action_space)]
        self.epsilon_decay = 1 / 50000
        self.gamma = GAMMA  # Discount
        self.eps0 = 1.0  # Epsilon greedy init
        self.model_path = model_path
        self.batch_size = BATCH_SIZE
        self.replay_memory = ReplayMemory(maxlen=REPLAY_MEMORY_SIZE)

        image_size = self.env.observation_space
        self.input_size = (image_size[0], image_size[1], image_size[2])
        self.output_size = self.env.action_space

        # Build both models
        self.model = self.build_model()
        self.target_model = self.build_model()
        # Make weights the same
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        """
        Construct the DQN model.
        """
        # image of size 8x8 with 3 channels (RGB)
        image_input = Input(shape=self.input_size)
        # preference weights
        weights_input = Input(shape=(NUM_WEIGHTS,))  # 6 weights

        # Define Layers
        x = image_input
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Concatenate()([x, weights_input])
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(4)(x)
        outputs = x

        # Build full model
        model = keras.Model(inputs=[image_input, weights_input], outputs=outputs)

        # Define optimizer and loss function
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        self.loss_fn = keras.losses.mean_squared_error

        return model

    def vanilla_epsilon_greedy(self, state, epsilon, weights, temperature=1.4, scene=None):
        """
        Select greedy action from model output based on current state with
        probability epsilon. With probability 1 - epsilon select random action.
        """
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model([state[np.newaxis], weights[np.newaxis]], training=False)
            return np.argmax(Q_values)

    def soft_policy(self, state, epsilon, weights, temperature=1.4, scene=None, truncated_thres=2):
        # print(f"weights:{weights}")
        q_values = self.model([state[np.newaxis], weights[np.newaxis]], training=False)
        q_values = tf.squeeze(q_values).numpy()
        max_2_index = q_values.argsort()[-truncated_thres:]
        mask = np.zeros(4)
        q_values -= np.mean(q_values)
        for i in max_2_index:
            mask[i] = 1
        q_masked = mask * q_values
        q_masked_sqr = q_masked ** temperature
        for i in range(len(q_masked_sqr)):
            if q_masked_sqr[i] == 0:
                q_masked_sqr[i] = -np.inf
        q_probs = softmax(q_masked_sqr)
        action = np.random.choice([0, 1, 2, 3], p=q_probs)
        # print(f"q_values:{q_values}\tq_probs:{q_probs}\taction:{action}")
        return action

    def normalize_pref(self, preference):
        sum_preference = sum(preference)
        normalized_preference = preference / sum_preference
        return normalized_preference

    def play_one_step(self, state, epsilon, weights):
        """
        Play one action using the DQN and store S A R S' in replay buffer.
        Adapted from:
            https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb
            [Accessed: 15/06/2020]
        """
        action = self.vanilla_epsilon_greedy(state, epsilon, weights)
        rewards, next_state, done, _ = self.env.step(action)
        next_state = np.float32(next_state)  # convert to float32 for tf
        reward = np.dot(rewards, weights)  # Linear scalarisation
        self.replay_memory.append((state, action, reward, next_state, done, weights))
        return next_state, reward, done, rewards

    def training_step(self):
        """
        Train the DQN on a batch from the replay buffer.
        Adapted from:
            https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb
            [Accessed: 15/06/2020]
        """
        # Sample a batch of S A R S' from replay memory
        experiences = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, weightss = experiences

        # Compute target Q values from 'next_states'
        next_Q_values = self.target_model([next_states, weightss], training=False)

        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)  # Make column vector

        # Mask to only consider action taken
        mask = tf.one_hot(actions, self.output_size)  # Number of actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            all_Q_values = self.model([states, weightss])
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train_model(self, episodes, preference, pref_space=None, save_per=100000, show_detail_per=10000):
        """
        Train the network over a range of episodes.
        """
        for episode in range(1, episodes + 1):
            rewards_vec = np.zeros(6)
            if not pref_space is None:
                preference = pref_space.sample()
            # normalized_preference = preference/np.sum(preference)
            # Decay epsilon
            eps = max(self.eps0 - episode * self.epsilon_decay, 0.01)

            # Reset env
            state = self.env.reset()
            state = np.float32(state)  # Convert to float32 for tf

            episode_reward = 0
            while True:
                # weights = self.normalize_pref(preference)
                state, reward, done, rewards = self.play_one_step(state, eps, preference)
                episode_reward += reward
                rewards_vec += rewards
                if done:
                    break
            if episode > START_TRAINING_AFTER:  # Wait for buffer to fill up a bit
                self.training_step()
                if episode % COPY_TO_TARGET_EVERY == 0:
                    self.target_model.set_weights(self.model.get_weights())
                if episode % save_per == 0 and episode >= save_per and self.model_path is not None:
                    self.model.save(self.model_path + str(episode))
            if episode % show_detail_per == 0:
                print(f"Epoch:{episode},\nEpoch Reward:{episode_reward},\nReward Vec:{rewards_vec},"
                      f"\nEpsilon:{eps},\nPreference:{preference}")
                print("---------------------------------")

    def play_episode(self, preference, deterministic_ratio=0):
        """
        Play one episode using the DQN and display the grid image at each step.
        """
        state = self.env.reset()
        # normalized_preference = preference/np.sum(preference)

        i = 0
        episode_reward = 0
        reward_vec = np.zeros(6)
        action_list = []
        while True:
            i += 1
            if np.random.rand() <= deterministic_ratio:
                action = self.vanilla_epsilon_greedy(state=state, epsilon=0, weights=preference,
                                                     scene=None)
            else:
                action = self.soft_policy(state=state, epsilon=0, weights=preference,
                                          temperature=TEMPERATURE, scene=None)
            rewards, state, done = self.env.step(action)
            state = np.float32(state)  # convert to float32 for tf
            # Add to stack
            reward = np.dot(rewards, preference)  # Linear scalarisation
            reward_vec = rewards + reward_vec * GAMMA
            # print("rewards:{},reward_vec:".format(rewards,reward_vec))
            episode_reward += reward
            action_list.append(action)
            # time.sleep(0.1)
            if done:
                return reward_vec, preference, action_list

    def generate_experience(self, episode, epsilon, weight_i, scene="train",
                            deterministic_ratio=1, num_trajs=1, temperature=1.4, max_state_traj_len=100,
                            reward_mask=True, truncated_thres=2):

        i = 0
        vec_r_traj = []
        action_traj = []
        states_traj = []
        while i < num_trajs:
            action_traj = []
            state_traj = np.zeros(64)
            state = self.env.reset()
            episode_reward = np.zeros(6)
            done = False
            while not done:
                if np.random.rand() <= deterministic_ratio:
                    action = self.vanilla_epsilon_greedy(state=state, epsilon=epsilon, weights=weight_i,
                                                         scene=scene)
                else:
                    action = self.soft_policy(state=state, epsilon=epsilon, weights=weight_i,
                                              temperature=temperature, scene=scene, truncated_thres=truncated_thres)

                rewards, next_state, done, position = self.env.step(action)
                state_traj[position] += 1
                episode_reward += rewards
                state = next_state
                action_traj.append(action)
            i += 1
            vec_r_traj.append(episode_reward)
            states_traj.append(state_traj)

        return vec_r_traj, action_traj, states_traj


if __name__ == '__main__':
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    start_time = datetime.now()
    model_path = "C://Users//19233436//PycharmProjects//DWPI//Environments//ItemGathering//ItemGatheringModel"
    item_gathering_env = ItemGathering()
    image = item_gathering_env.reset()
    preference_space = PreferenceSpace()
    dqn_ag = DQNAgent(item_gathering_env, model_path=model_path)
    dqn_ag.train_model(TRAINING_EPISODES, pref_space=preference_space, preference=None, save_per=50000,
                       show_detail_per=1000)
    run_time = datetime.now() - start_time
    print(f'Run time: {run_time} s')
