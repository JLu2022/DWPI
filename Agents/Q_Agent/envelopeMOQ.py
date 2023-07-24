import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from keras.models import load_model
from keras.layers import Concatenate
from keras.layers import Dense
from datetime import datetime
from Agents.Q_Agent.utils.preference_space import PreferenceSpace
from Agents.Q_Agent.utils.replay_memory import ReplayMemory
from Environments.DeepSeaTreasure.ImageDST import ImageDST
import matplotlib.pyplot as plt
from Agents.Q_Agent.utils.model_construct import QNN
from keras.losses import mse, huber, mae
from keras.optimizers import Adam
from scipy.special import softmax

# SEED = 16

BATCH_SIZE = 64
PREFERENCE_BATCH_SIZE = 10
GAMMA = 1

NUM_EPOCHS = 10
NUM_EPISODES = 3000
UPDATE_TARGET_PER = 200  # Steps
START_TRAINING_AFTER = 20  # Episodes
SHOW_DETAIL_PER = 1000
NUM_WEIGHTS = 2
default_pref = np.array([0.93, 0.07])
ACTIONS = {0: "上", 1: "下", 2: "左", 3: "右"}


class EnvelopeMOQ:

    def __init__(self, env, model_path=None, replay_memory=None, preference_space=None, pre_load_model=None,
                 mode="vec_envelope"):
        self.env = env
        self.actions = range(self.env.action_space)

        self.model_path = model_path
        self.batch_size = BATCH_SIZE
        self.replay_memory = replay_memory
        self.preference_space = preference_space

        image_size = self.env.observation_space
        self.input_size = (image_size[0], image_size[1], image_size[2])

        img_input = Input(shape=self.input_size)
        header = keras.models.load_model("C://Users//19233436//PycharmProjects//DWPI//Agents//AE_model//E_model")
        header.trainable = False
        x = header.layers[2](img_input)
        x = header.layers[3](x)
        x = header.layers[4](x)
        x = header.layers[5](x)
        x = header.layers[6](x)
        eye_output = header.layers[7](x)
        self.state_space = 25
        self.obs_to_state = keras.Model(inputs=img_input, outputs=eye_output)
        self.obs_to_state.summary()

        self.QNN_time = QNN(hidden_dim=16, name="QNN_time", out_dim=self.env.action_space)
        self.target_QNN_time = QNN(hidden_dim=16, name="target_QNN_time", out_dim=self.env.action_space)
        self.target_QNN_time.set_weights(self.QNN_time.get_weights())

        self.QNN_treasure = QNN(hidden_dim=16, name="QNN_treasure", out_dim=self.env.action_space)
        self.target_QNN_treasure = QNN(hidden_dim=16, name="target_QNN_treasure", out_dim=self.env.action_space)
        self.target_QNN_treasure.set_weights(self.QNN_treasure.get_weights())

        self.vanilla_QNN = QNN(hidden_dim=16, name="QNN_vanilla", out_dim=self.env.action_space)
        self.target_vanilla_QNN = QNN(hidden_dim=16, name="target_vanilla_QNN", out_dim=self.env.action_space)
        self.target_vanilla_QNN.set_weights(self.vanilla_QNN.get_weights())

        self.vec_envelope_QNN = QNN(hidden_dim=16, name="vec_envelope_QNN", out_dim=self.env.action_space * 2)
        self.target_vec_envelope_QNN = QNN(hidden_dim=16, name="target_vec_envelope_QNN",
                                           out_dim=self.env.action_space * 2)
        self.target_vec_envelope_QNN.set_weights(self.vec_envelope_QNN.get_weights())

        self.time_optimizer = Adam(learning_rate=1e-3)
        self.treasure_optimizer = Adam(learning_rate=1e-3)
        self.vanilla_QNN_optimizer = Adam(learning_rate=1e-3)
        self.vec_envelope_QNN_optimizer = Adam(learning_rate=1e-3)

    def vec_envelope_epsilon_greedy(self, state, epsilon, weight_i, scene="train"):
        if np.random.rand() < epsilon:
            return random.choice(self.actions), " (rand) "
        else:
            s_w_i = np.insert(state, [self.state_space, self.state_space], values=weight_i, axis=1)

            Q_vec = np.squeeze(self.vec_envelope_QNN(s_w_i, training=False).numpy())
            Q = weight_i[0] * Q_vec[:4] + weight_i[1] * Q_vec[4:]
            action = np.argmax(Q)

            return action, " (policy) "

    def vanilla_epsilon_greedy(self, state, epsilon, weight_i, scene="train"):
        if np.random.rand() < epsilon:
            return random.choice(self.actions), " (rand) "
        else:
            s_w_i = np.insert(state, [self.state_space, self.state_space], values=weight_i, axis=1)
            Q_values = np.squeeze(self.vanilla_QNN(s_w_i, training=False).numpy())
            action = np.argmax(Q_values)
            return action, " (policy) "

    def envelope_epsilon_greedy(self, state, epsilon, weight_i, scene="train"):
        if np.random.rand() < epsilon:
            return random.choice(self.actions), " (rand) "
        else:
            s_w_i = np.insert(state, [self.state_space, self.state_space], values=weight_i, axis=1)
            Q_time = np.squeeze(self.QNN_time(s_w_i, training=False).numpy())
            Q_treasure = np.squeeze(self.QNN_treasure(s_w_i, training=False).numpy())

            Q_values = weight_i[0] * Q_time + weight_i[1] * Q_treasure
            action = np.argmax(Q_values)
            return action, " (policy) "

    def policy_truncated_softmax(self, state, weights, temperature=1.5):
        Q_time = np.squeeze(self.QNN_time(state, training=False).numpy())
        Q_treasure = np.squeeze(self.QNN_treasure(state, training=False).numpy())
        Q_values = np.sum(weights * np.array([Q_time, Q_treasure]).T, axis=1)
        Q_values = np.squeeze(Q_values)
        max_2_index = Q_values.argsort()[-2:]
        mask = np.zeros(4)
        Q_values -= np.mean(Q_values)
        for i in max_2_index:
            mask[i] = 1

        q_masked = mask * Q_values
        q_masked_sqr = q_masked ** temperature
        for i in range(len(q_masked_sqr)):
            if q_masked_sqr[i] == 0:
                q_masked_sqr[i] = -np.inf
        q_probs = softmax(q_masked_sqr)
        action = np.random.choice([0, 1, 2, 3], p=q_probs)
        return action, Q_values, q_probs, state

    def train(self, episode=0, tau=0.2, weight_i=None, mode="envelope",
              preference_threshold=1.0):  # target_update_coefficient is tau
        if mode == "envelope":
            self.envelope_training_step(episode, weight_i, preference_threshold)
            if episode % UPDATE_TARGET_PER == 0:
                self.target_QNN_time.set_weights(self.QNN_time.get_weights())
                self.target_QNN_treasure.set_weights(self.QNN_treasure.get_weights())
        if mode == "vanilla":
            self.vanilla_training_step(episode, weight_i, preference_threshold)
            if episode % UPDATE_TARGET_PER == 0:
                self.target_vanilla_QNN.set_weights(self.vanilla_QNN.get_weights())
        if mode == "vec_envelope":
            self.vec_envelope_training_step(episode, weight_i, sample_pref=True,
                                            preference_threshold=preference_threshold)
            if episode % UPDATE_TARGET_PER == 0:
                self.target_vec_envelope_QNN.set_weights(self.vec_envelope_QNN.get_weights())

    def vec_envelope_training_step(self, episode=0, weight_i=None, sample_pref=True, preference_threshold=1.0):
        experiences = self.replay_memory.sample(BATCH_SIZE)
        states, actions, rewards_time, rewards_treasure, rewards_scalar, next_states, dones, weightss = experiences
        # ======================================================================================================#
        if sample_pref:
            weights = self.preference_space.sample_batch(batch_size=PREFERENCE_BATCH_SIZE,
                                                         threshold=preference_threshold)
            next_Q_time_tensor = []
            next_Q_treasure_tensor = []
            next_Q_tensor = []
            for w_prime in weights:
                n_s_w_prime = np.insert(next_states, [self.state_space, self.state_space], values=weight_i, axis=1)
                next_Q_vecs_prime = self.target_vec_envelope_QNN(n_s_w_prime, training=False)
                next_time_Qs_prime = next_Q_vecs_prime[:, :4]
                next_treasure_Qs_prime = next_Q_vecs_prime[:, 4:]
                next_Qs_prime = w_prime[0] * next_time_Qs_prime + w_prime[1] * next_treasure_Qs_prime
                indices = np.argmax(next_Qs_prime, axis=1)
                next_Q_mask = np.eye(self.env.action_space)[indices]

                max_next_Q_prime = np.sum(next_Qs_prime * next_Q_mask, axis=1).flatten()
                max_next_Q_time_prime = np.sum(next_time_Qs_prime * next_Q_mask, axis=1).flatten()
                max_next_Q_treasure_prime = np.sum(next_treasure_Qs_prime * next_Q_mask, axis=1).flatten()

                next_Q_time_tensor.append(max_next_Q_time_prime)
                next_Q_treasure_tensor.append(max_next_Q_treasure_prime)
                next_Q_tensor.append(max_next_Q_prime)

            next_Q_time_tensor = np.array(next_Q_time_tensor)
            next_Q_treasure_tensor = np.array(next_Q_treasure_tensor)
            next_Q_tensor = np.array(next_Q_tensor)

            next_max_Q_indices = np.argmax(next_Q_tensor, axis=0)
            preference_mask = np.eye(PREFERENCE_BATCH_SIZE)[next_max_Q_indices]

            # max_next_Q = np.sum(next_Q_tensor.T * preference_mask, axis=1)
            max_next_Q_time = np.sum(next_Q_time_tensor.T * preference_mask, axis=1)
            max_next_Q_treasure = np.sum(next_Q_treasure_tensor.T * preference_mask, axis=1)

            target_Q_time = (rewards_time + (1 - dones) * GAMMA * max_next_Q_time)
            target_Q_treasure = (rewards_treasure + (1 - dones) * GAMMA * max_next_Q_treasure)
            Q_mask = np.eye(self.env.action_space)[actions]
        # ======================================================================================================#
        else:
            n_s_w_i = np.insert(next_states, [self.state_space, self.state_space], values=weight_i, axis=1)
            next_Q_vecs = self.target_vec_envelope_QNN(n_s_w_i, training=False)
            next_time_Qs = next_Q_vecs[:, :4]
            next_treasure_Qs = next_Q_vecs[:, 4:]
            next_Qs = weight_i[0] * next_time_Qs + weight_i[1] * next_treasure_Qs

            indices = np.argmax(next_Qs, axis=1)
            next_Q_mask = np.eye(self.env.action_space)[indices]

            max_next_Q_time = np.sum(next_time_Qs * next_Q_mask, axis=1).flatten()
            max_next_Q_treasure = np.sum(next_treasure_Qs * next_Q_mask, axis=1).flatten()

            target_Q_time = (rewards_time + (1 - dones) * GAMMA * max_next_Q_time)
            target_Q_treasure = (rewards_treasure + (1 - dones) * GAMMA * max_next_Q_treasure)
            Q_mask = np.eye(self.env.action_space)[actions]
        # factor = min(episode / (NUM_EPISODES * NUM_EPOCHS), 0.8)
        factor = episode / (NUM_EPISODES * NUM_EPOCHS)
        s_w_i = np.insert(states, [self.state_space, self.state_space], values=weight_i, axis=1)
        with tf.GradientTape() as tape:
            Q_vecs = self.vec_envelope_QNN(s_w_i)
            time_Q_vecs = Q_vecs[:, :4]
            treasure_Q_vecs = Q_vecs[:, 4:]
            masked_time_Qs = tf.squeeze(time_Q_vecs) * Q_mask
            masked_treasure_Qs = tf.squeeze(treasure_Q_vecs) * Q_mask
            time_Qs = tf.reduce_sum(masked_time_Qs, axis=1)
            treasure_Qs = tf.reduce_sum(masked_treasure_Qs, axis=1)
            y = tf.transpose(tf.stack([target_Q_time, target_Q_treasure]))
            Q_hat = tf.transpose(tf.stack([time_Qs, treasure_Qs]))
            loss_A = tf.reduce_mean(mse(y, Q_hat))
            w_T_y = weight_i[0] * target_Q_time + weight_i[1] * target_Q_treasure
            w_T_Q_hat = weight_i[0] * time_Qs + weight_i[1] * treasure_Qs
            loss_B = tf.reduce_mean(mae(w_T_y, w_T_Q_hat))
            loss = loss_A * (1 - factor) + loss_B * factor

        grad = tape.gradient(loss, self.vec_envelope_QNN.trainable_variables)
        self.vec_envelope_QNN_optimizer.apply_gradients(zip(grad, self.vec_envelope_QNN.trainable_variables))
        if episode % SHOW_DETAIL_PER == 0:
            print(f"loss:\t{np.round(loss, 2)}\n"
                  f"loss_A:\t{np.round(loss_A, 2)}\n"
                  f"loss_B:\t{np.round(loss_B, 2)}")

    def vanilla_training_step(self, episode=0, weight_i=None, preference_threshold=1.0):
        experiences = self.replay_memory.sample(BATCH_SIZE)
        states, actions, rewards_time, rewards_treasure, rewards_scalar, next_states, dones, weightss = experiences

        n_s_w_i = np.insert(next_states, [self.state_space, self.state_space], values=weight_i, axis=1)
        next_Qs = self.target_vanilla_QNN(n_s_w_i, training=False)

        indices = np.argmax(next_Qs, axis=1)
        next_Q_mask = np.eye(self.env.action_space)[indices]

        max_next_Q = np.sum(next_Qs * next_Q_mask, axis=1).flatten()
        target_Q = (rewards_scalar + (1 - dones) * GAMMA * max_next_Q)
        # print(f"target_Q:\n{target_Q}")
        Q_mask = np.eye(self.env.action_space)[actions]

        s_w_i = np.insert(states, [self.state_space, self.state_space], values=weight_i, axis=1)
        with tf.GradientTape() as tape:
            Qs = self.vanilla_QNN(s_w_i)
            masked_Q = tf.squeeze(Qs) * Q_mask
            Q = tf.reduce_sum(masked_Q, axis=1)
            # print(f"masked_Q:\n{Q}")
            TD_loss = mse(target_Q, Q)

        grad = tape.gradient(TD_loss, self.vanilla_QNN.trainable_variables)
        # print(
        #     # f"gradient:\n{grad}\n"
        #       f"gradient:\n{grad[0]}\n")
        self.vanilla_QNN_optimizer.apply_gradients(zip(grad, self.vanilla_QNN.trainable_variables))
        if episode % SHOW_DETAIL_PER == 0:
            print(f"TD_loss:{np.round(TD_loss, 2)}\n")

    def envelope_training_step(self, episode=0, weight_i=None, preference_threshold=1.0):

        experiences = self.replay_memory.sample(BATCH_SIZE)
        weights = self.preference_space.sample_batch(batch_size=PREFERENCE_BATCH_SIZE, threshold=preference_threshold)

        states, actions, rewards_time, rewards_treasure, rewards_scalar, next_states, dones, weightss = experiences

        next_Q_time_tensor = []
        next_Q_treasure_tensor = []
        next_Q_tensor = []

        for weight_prime in weights:
            n_s_w_prime = np.insert(next_states, [self.state_space, self.state_space], values=weight_prime, axis=1)
            # print(f"weight_prime:{weight_prime}\nnext_states:{next_states}\nn_s_w_prime:{n_s_w_prime}")
            next_time_Qs = self.target_QNN_time(n_s_w_prime, training=False)
            next_treasure_Qs = self.target_QNN_treasure(n_s_w_prime, training=False)
            next_weighted_time_Qs = next_time_Qs * weight_prime[0]
            next_weighted_treasure_Qs = next_treasure_Qs * weight_prime[1]
            next_Qs = next_weighted_treasure_Qs + next_weighted_time_Qs

            indices = np.argmax(next_Qs, axis=1)
            next_Q_mask = np.eye(self.env.action_space)[indices]

            max_next_Q = np.sum(next_Qs * next_Q_mask, axis=1).flatten()
            max_next_Q_time = np.sum(next_time_Qs * next_Q_mask, axis=1).flatten()
            max_next_Q_treasure = np.sum(next_treasure_Qs * next_Q_mask, axis=1).flatten()

            next_Q_time_tensor.append(max_next_Q_time)
            next_Q_treasure_tensor.append(max_next_Q_treasure)
            next_Q_tensor.append(max_next_Q)

        next_Q_time_tensor = np.array(next_Q_time_tensor)
        next_Q_treasure_tensor = np.array(next_Q_treasure_tensor)
        next_Q_tensor = np.array(next_Q_tensor)

        next_max_Q_indices = np.argmax(next_Q_tensor, axis=0)
        preference_mask = np.eye(PREFERENCE_BATCH_SIZE)[next_max_Q_indices]
        max_next_Q = np.sum(next_Q_tensor.T * preference_mask, axis=1)
        max_next_Q_time = np.sum(next_Q_time_tensor.T * preference_mask, axis=1)
        max_next_Q_treasure = np.sum(next_Q_treasure_tensor.T * preference_mask, axis=1)

        target_Q_time = (rewards_time + (1 - dones) * GAMMA * max_next_Q_time)
        target_Q_treasure = (rewards_treasure + (1 - dones) * GAMMA * max_next_Q_treasure)
        Q_mask = np.eye(self.env.action_space)[actions]

        factor_1 = episode / NUM_EPISODES
        s_w_i = np.insert(states, [self.state_space, self.state_space], values=weight_i, axis=1)
        with tf.GradientTape(persistent=True) as tape:

            Qs_time = self.QNN_time(s_w_i)
            Qs_treasure = self.QNN_treasure(s_w_i)

            masked_Q_time_vec = tf.squeeze(Qs_time) * Q_mask
            masked_Q_treasure_vec = tf.squeeze(Qs_treasure) * Q_mask

            Q_time = tf.reduce_sum(masked_Q_time_vec, axis=1)
            Q_treasure = tf.reduce_sum(masked_Q_treasure_vec, axis=1)

            y = tf.transpose(tf.stack([target_Q_time, target_Q_treasure]))
            Q_vec = tf.transpose(tf.stack([Q_time, Q_treasure]))

            loss_A_time = mse(Q_time, target_Q_time)
            loss_A_treasure = mse(Q_treasure, target_Q_treasure)

            w_T_y = tf.reduce_mean(tf.multiply(y, weight_i), axis=1)
            w_T_Q = tf.reduce_mean(tf.multiply(Q_vec, weight_i), axis=1)
            loss_B = tf.reduce_mean(mae(w_T_y, w_T_Q))

            loss_B = tf.dtypes.cast(loss_B, tf.float64)

            loss_time = (1 - factor_1) * loss_A_time + factor_1 * loss_B
            loss_treasure = (1 - factor_1) * loss_A_treasure + factor_1 * loss_B

        grad_time = tape.gradient(loss_time, self.QNN_time.trainable_variables)
        self.time_optimizer.apply_gradients(zip(grad_time, self.QNN_time.trainable_variables))

        grad_treasure = tape.gradient(loss_treasure, self.QNN_treasure.trainable_variables)
        self.treasure_optimizer.apply_gradients(zip(grad_treasure, self.QNN_treasure.trainable_variables))
        del tape

        if episode % SHOW_DETAIL_PER == 0:
            print(
                # f"TD_loss_time:{np.round(TD_loss_time, 2)}\n"
                #   f"TD_loss_treasure:{np.round(TD_loss_treasure, 2)}\n"
                f"loss_time:\t{np.round(loss_time, 2)}\n"
                f"loss_A_time:\t{loss_A_time}\n"
                f"loss_treasure:\t{np.round(loss_treasure, 2)}\n"
                f"loss_A_treasure:\t{loss_A_treasure}\n"
                f"loss_B:\t{np.round(loss_B, 2)}"
                # f"loss treasure:{np.round(loss_treasure, 2)}"
            )
        # return _, _, _

    def generate_experience(self, episode, epsilon, weight_i, mode="envelope", scene="train"):
        obs = self.env.reset()
        state = self.obs_to_state(np.expand_dims(obs, axis=0))
        reward_result = np.zeros(2)
        action_traj = ""
        split = "-"
        utility = 0

        while True:
            if mode == "envelope":
                action, token = self.envelope_epsilon_greedy(state, epsilon, weight_i, scene=scene)
            elif mode == "vanilla":
                action, token = self.vanilla_epsilon_greedy(state, epsilon, weight_i, scene=scene)
            elif mode == "vec_envelope":
                action, token = self.vec_envelope_epsilon_greedy(state, epsilon, weight_i, scene=scene)
            else:
                action, token = self.vanilla_epsilon_greedy(state, epsilon=0, weight_i=weight_i, scene=scene)

            rewards, next_obs, done = self.env.step(action, episode=episode)
            next_state = self.obs_to_state(np.expand_dims(next_obs, axis=0))

            reward_time = rewards[0]
            reward_treasure = rewards[1]
            reward_scalar = reward_time * weight_i[0] + reward_treasure * weight_i[1]
            self.replay_memory.append(
                (state[0], action, reward_time, reward_treasure, reward_scalar, next_state[0], done, weight_i))

            state = next_state
            reward_result += np.array([reward_time, reward_treasure])
            utility += reward_time * weight_i[0] + reward_treasure * weight_i[1]

            action_traj += ACTIONS[action] + token + split
            if done:
                break

        if episode % SHOW_DETAIL_PER == 0:
            print(f"R treasure:{reward_result}\n"
                  # f"utility:{utility}\n"
                  f"Action traj:{action_traj[:-1]}\n"
                  f"weight:{weight_i}")
            print("----------------------")
        return reward_result

    def save_model(self, path, mode="envelope"):
        if mode == "envelope":
            self.QNN_time.save(path + "QNN_time//")
            self.QNN_treasure.save(path + "QNN_treasure//")
        if mode == "vanilla":
            self.vanilla_QNN.save(path + "vanilla_QNN//")
        if mode == "vec_envelope":
            self.vec_envelope_QNN.save(path + "vec_envelope_QNN//")

    def load_model(self, path, mode="envelope"):

        if mode == "envelope":
            input_layer_0 = Input(shape=27)
            input_layer_1 = Input(shape=27)
            self.QNN_time = load_model(path + "QNN_time//")
            self.QNN_time = self.QNN_time(input_layer_0)
            self.QNN_time = keras.Model(inputs=input_layer_0, outputs=self.QNN_time)
            self.QNN_treasure = load_model(path + "QNN_treasure//")
            self.QNN_treasure = self.QNN_treasure(input_layer_1)
            self.QNN_treasure = keras.Model(inputs=input_layer_1, outputs=self.QNN_treasure)
        if mode == "vanilla":
            input_layer = Input(shape=27)
            self.vanilla_QNN = load_model(path + "vanilla_QNN//")
            self.vanilla_QNN = self.vanilla_QNN(input_layer)
            self.vanilla_QNN = keras.Model(inputs=input_layer, outputs=self.vanilla_QNN)
        if mode == "vec_envelope":
            input_layer = Input(shape=27)
            self.vec_envelope_QNN = load_model(path + "vec_envelope_QNN//")
            self.vec_envelope_QNN = self.vec_envelope_QNN(input_layer)
            self.vec_envelope_QNN = keras.Model(inputs=input_layer, outputs=self.vec_envelope_QNN)


if __name__ == '__main__':
    # np.random.seed(SEED)
    # tf.random.set_seed(SEED)
    # random.seed(SEED)
    start_time = datetime.now()

    imgDST = ImageDST()
    image = imgDST.reset()
    pref_space = PreferenceSpace(granularity=100)
    replay_memory = ReplayMemory(maxlen=20000)
    model_path = "C://Users//19233436//PycharmProjects//DWPI//Agents//Q_Agent//Envelope_Q_model//"
    dqn_ag = EnvelopeMOQ(imgDST,
                         model_path=model_path,
                         preference_space=pref_space,
                         replay_memory=replay_memory)

    episode_reward = np.zeros(2)
    preference_upper_bound = 0.34
    preference_lower_bound = 0.05
    mode = "envelope"

    # dqn_ag.load_model(path=model_path,
    #                   mode=mode)
    weight_i = pref_space.sample(default_pref=default_pref)
    for epoch in range(1, NUM_EPOCHS + 1):
        weight_i = pref_space.sample()
        # while weight_i[1] > preference_upper_bound or weight_i[1] < preference_lower_bound:
        #     weight_i = pref_space.sample()
        for episode in range(1, NUM_EPISODES + 1):
            epsilon = max((NUM_EPISODES - episode) / NUM_EPISODES, 0.1)

            if episode % SHOW_DETAIL_PER == 0:
                print("==================================================")
                episode_reward /= SHOW_DETAIL_PER
                print(f"Episodes:{epoch}-{episode}\n"
                      f"Epsilon:{epsilon}\n"
                      f"Weight for this episode:{weight_i}\n"
                      f"mean episode rewards:{episode_reward}")
                episode_reward = np.zeros(2)

            reward_vec = dqn_ag.generate_experience(epsilon=epsilon,
                                                    episode=episode,
                                                    weight_i=weight_i,
                                                    mode=mode
                                                    # , scene="evaluate"
                                                    )
            episode_reward += reward_vec
            if episode > START_TRAINING_AFTER:
                dqn_ag.train(episode=episode, tau=1, weight_i=weight_i, mode=mode, preference_threshold=0.11)

    dqn_ag.save_model(path=model_path,
                      mode=mode)
    dqn_ag.load_model(path=model_path,
                      mode=mode)
    for p in range(101):
        treasure_pref = p / 100
        dqn_ag.generate_experience(epsilon=0,
                                   episode=0,
                                   weight_i=np.array([1 - treasure_pref, treasure_pref]),
                                   mode=mode,
                                   scene="evaluate")
    run_time = datetime.now() - start_time
    print(f'Run time: {run_time} s')
