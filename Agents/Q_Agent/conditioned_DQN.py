import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Dense
from datetime import datetime
from Agents.Q_Agent.utils.preference_space import PreferenceSpace
from Agents.Q_Agent.utils.replay_memory import ReplayMemory
from Environments.DeepSeaTreasure.ImageDST import ImageDST
import matplotlib.pyplot as plt

SEED = 42

REPLAY_MEMORY_SIZE = 20000
BATCH_SIZE = 64

GAMMA = 1

TRAINING_EPISODES = 50000

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1 / (1000 * 0.98)
NEAR_ZERO = 1e-10
COPY_TO_TARGET_EVERY = 200  # Steps
START_TRAINING_AFTER = 100  # Episodes

FRAME_STACK_SIZE = 3
NUM_WEIGHTS = 2


class DQNAgent:

    def __init__(self, env, model_path=None, checkpoint=True, replay_memory=None):
        self.env = env
        self.actions = range(self.env.action_space)

        self.gamma = GAMMA  # Discount
        self.eps0 = 1.0  # Epsilon greedy init
        self.model_path = model_path
        self.batch_size = BATCH_SIZE
        self.replay_memory = replay_memory
        self.checkpoint = checkpoint
        image_size = self.env.observation_space

        self.input_size = (image_size[0], image_size[1], image_size[2])
        self.output_size = self.env.action_space

        # Build both models
        self.model = self.build_model()
        self.target_model = self.build_model()
        # Make weights the same
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        weights_input = Input(shape=(NUM_WEIGHTS,))  # 6 weights
        img_input = Input(shape=self.input_size)
        header = keras.models.load_model("C://Users//19233436//PycharmProjects//DWPI//Agents//AE_model//E_model")
        header.trainable = False
        x = header.layers[2](img_input)
        x = header.layers[3](x)
        x = header.layers[4](x)
        x = header.layers[5](x)
        x = header.layers[6](x)
        x = header.layers[7](x)
        x = Concatenate()([x, weights_input])
        x = Dense(16, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(self.output_size)(x)
        outputs = x

        # Build full model
        model = keras.Model(inputs=[img_input, weights_input], outputs=outputs)

        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = keras.losses.mean_squared_error
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        # model.summary()
        return model

    def epsilon_greedy_policy(self, state, epsilon, weights, episode=0):
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model([state[np.newaxis], weights[np.newaxis]], training=False)
            Q_values = self.symexp(Q_values)
            action = np.argmax(Q_values)
            return action

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def play_one_step(self, state, epsilon, preference, episode=10):
        action = self.epsilon_greedy_policy(state, epsilon, preference, episode)
        rewards, next_state, done,_ = self.env.step(action, episode=episode)
        next_state = np.float32(next_state)  # convert to float32 for tf
        print(f"next_state shape:{next_state.shape}")
        reward = np.dot(rewards, preference)  # Linear scalarisation
        self.replay_memory.append((state, action, rewards[0], rewards[1], reward, next_state, done, preference))
        return next_state, reward, done, rewards

    def symlog(self, x):
        # if (x == 0).any():
        x = x + NEAR_ZERO
        sign = x / np.abs(x)
        symlog_x = sign * np.log(np.abs(x) + 1)
        return symlog_x

    def symexp(self, x):
        # if (x == 0).any():
        x = x + NEAR_ZERO
        sign = x / np.abs(x)
        symexp_x = sign * (np.exp(np.abs(x)) - 1)
        return symexp_x

    def training_step(self, current_weight=None, current_state=None):
        experiences = self.replay_memory.sample(self.batch_size, current_weight=current_weight,
                                                max_batch_size=100, NER_ratio=0.8)
        states, actions, time_rewards, treasure_rewards, reward_scalar, next_states, dones, weightss = experiences

        # Compute target Q values from 'next_states'
        print(f"next_states shape:{next_states.shape}")
        next_Q_values = self.target_model([next_states, weightss], training=False)
        print(f"next_Q_values shape:{next_Q_values.shape}")
        max_next_Q_values = np.max(next_Q_values, axis=1)
        print(f"max_next_Q_values:{max_next_Q_values.shape}")
        # print(
        #     f"max_next_Q_values:{max_next_Q_values}\n----------------------------------------\nrewards:{reward_scalar}")
        target_Q_values = (reward_scalar + (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)  # Make column vector
        target_Q_values = self.symlog(target_Q_values)
        mask = tf.one_hot(actions, self.output_size)  # Number of actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            all_Q_values = self.model([states, weightss])

            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            Q_values = self.symlog(Q_values)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def generate_experience(self, episode, epsilon, preference):
        state = self.env.reset()
        while True:
            state, reward, done, rewards = self.play_one_step(state, epsilon, preference, episode=episode)
            if done:
                break

    def train(self, episode=0, save_per=100, preference=None):

        if episode > START_TRAINING_AFTER:
            self.training_step(current_weight=preference)

            if episode % COPY_TO_TARGET_EVERY == 0:
                self.target_model.set_weights(self.model.get_weights())

            if episode % save_per == 0 and episode >= save_per and self.checkpoint:
                self.model.save(self.model_path + str(episode))

    def show_start_img(self):
        state = self.env.reset()
        # print(f"position:{self.initial_position}")
        plt.imshow(state)
        plt.show()
        plt.close()

    def conventional_train(self, num_episodes, pref_space, save_per=100000, show_detail=True, trainable=True):

        for episode in range(1, num_episodes + 1):
            rewards_vec = np.zeros(2)
            preference = pref_space.sample()
            eps = max(self.eps0 - episode * (1 / (num_episodes * 0.98)), EPSILON_END)

            state = self.env.reset()
            state = np.float32(state)  # Convert to float32 for tf
            episode_reward = 0

            while True:
                state, reward, done, rewards = self.play_one_step(state, eps, preference, episode=episode)
                episode_reward += reward
                rewards_vec += rewards
                if done:
                    break

            if episode > START_TRAINING_AFTER:
                for _ in range(1):
                    self.training_step(current_weight=preference)

                if episode % COPY_TO_TARGET_EVERY == 0:
                    self.target_model.set_weights(self.model.get_weights())

                if episode % save_per == 0 and episode >= save_per and self.checkpoint:
                    self.model.save(self.model_path + str(episode))

            if show_detail and episode % 500 == 0:
                print("Epoch:{},\tEpoch Reward:{},\tReward Vec:{},\tEpsilon:{},\tPreference:{}".format(
                    episode, episode_reward, rewards_vec, eps, preference))

    def play_episode(self, preference):
        """
        Play one episode using the DQN and display the grid image at each step.
        """
        state = self.env.reset()
        i = 0
        episode_reward = 0
        reward_vec = np.zeros(2)
        action_list = []
        while True:
            i += 1
            action = self.epsilon_greedy_policy(state, 0.0, preference)
            rewards, state, done = self.env.step(action)
            state = np.float32(state)  # convert to float32 for tf
            reward = np.dot(rewards, preference)  # Linear scalarisation
            reward_vec += rewards
            episode_reward += reward
            plt.imshow(state)
            plt.show(block=True)
            plt.pause(0.4)
            plt.close()
            action_list.append(action)
            if done:
                # print("rewards:", rewards)
                return reward_vec, preference, action_list
                # break


if __name__ == '__main__':
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    start_time = datetime.now()

    imgDST = ImageDST()
    image = imgDST.reset()
    pref_space = PreferenceSpace()
    replay_memory = ReplayMemory(maxlen=2000)
    dqn_ag = DQNAgent(imgDST, model_path="C://PhD//2023//Inverse_MORL//MOA2C//conditioned_DQN//",
                      replay_memory=replay_memory)
    # print(dqn_ag.symlog(np.array([-1,1,3])))
    dqn_ag.conventional_train(TRAINING_EPISODES, pref_space, save_per=10000)
    dqn_ag.model = keras.models.load_model("C://PhD//2023//Inverse_MORL//MOA2C//conditioned_DQN//50000//")
    dqn_ag.play_episode(preference=np.array([0.1, 0.9]))
    run_time = datetime.now() - start_time
    print(f'Run time: {run_time} s')
