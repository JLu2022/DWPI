import numpy as np
import scipy.stats

from mo_traffic_agent import PreferenceSpace, DQNAgent, MOTraffic
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow import keras
import keras
from keras import layers, backend as K
from keras.losses import mse, losses_utils  # osineSimilarity, cosine_similarity, mean_absolute_error
from datetime import datetime  # Used for timing script


class Inferrer(object):
    def __init__(self, actor):
        self.inferring_model = self.build_model()
        self.actor = actor

    def build_model(self):
        input_dim = 5
        output_dim = 5

        original_inputs = tf.keras.Input(shape=(input_dim,), name="encoder_input")
        x = layers.Dense(32, activation="relu")(original_inputs)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        z = layers.Dense(output_dim, name="z", activation="softmax")(x)

        model = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        model.compile(optimizer, loss=mse, run_eagerly=True)
        model.summary()
        return model

    def train(self, x, y, batch_size=64, epochs=3000, monitor="loss", factor=0.8, patience=20, min_lr=1e-6):
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, min_lr=min_lr)]
        train_history = self.inferring_model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        return train_history

    def load_model(self, path):
        self.inferring_model.load_weights(path)

    def save_model(self, path):
        self.inferring_model.save_weights(path)


def min_max_normalize(min_v, max_v, target):
    return (target - min_v) / (max_v - min_v)


def sum_normalize(input_list):
    sum_value = sum(input_list)
    input_list = np.array(input_list)
    output_list = np.round(input_list / sum_value, 2)
    return output_list


def iterative_show(img):
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(5)
    plt.close()


if __name__ == '__main__':
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    # For timing the script
    start_time = datetime.now()
    traffic_env = MOTraffic()

    # Initialise Preference Weight Space
    pref_space = PreferenceSpace()

    # Instantiate agent (pass in environment)
    dqn_ag = DQNAgent(traffic_env, model_path=None)
    dqn_ag.model = keras.models.load_model(
        "C://Users//Junlin//Inverse_MORL//Experiment//Traffic//Traffic_Model//200000")
    dqn_ag.target_model = keras.models.load_model(
        "C://Users//Junlin//Inverse_MORL//Experiment//Traffic//Traffic_Model//200000")

    inferrer = Inferrer(actor=dqn_ag)
    raw_x_train_reward = np.load("C://Users//Junlin//Inverse_MORL//Experiment//Traffic//Dataset//x_train_raw.npy")
    raw_y_train_preference = np.load("C://Users//Junlin//Inverse_MORL//Experiment//Traffic//Dataset//y_train_raw.npy")
    y_train = []
    for y in raw_y_train_preference:
        y_train.append(dqn_ag.normalize_pref(y))
    y_train = np.array(y_train)

    max_value = -np.inf
    min_value = np.inf
    for x in raw_x_train_reward:
        if max_value < max(x):
            max_value = max(x)
        if min_value > min(x):
            min_value = min(x)
    # x_train = min_max_normalize(min_value, max_value, raw_x_train_reward)
    x_train = raw_x_train_reward
    start_time = datetime.now()
    print(x_train,"\n",y_train)
    # inferrer.train(x_train, y_train, batch_size=64)
    run_time = datetime.now() - start_time
    print(f'Train Time DST: {run_time} s')
    # inferrer.save_model("C://Users//Junlin//Inverse_MORL//Experiment//Traffic//Inferrer_Model")

    # LOAD the model and do inference
    inferrer.load_model("C://Users//Junlin//Inverse_MORL//Experiment//Traffic//Inferrer_Model")
    ground_truth_dict = {"Always_Safe": np.array([1, 50, 10, 50, 1]),
                         "Always_Fast": np.array([10, 50, 10, 10, 1]),
                         "Fast_Safe": np.array([5, 50, 0, 50, 1]),
                         "Slow_Safe": np.array([1, 50, 0, 50, 1])}
    start_time = datetime.now()
    # for x_entry in raw_x_train:
    #     traffic_env.reset()
    #     reward_vec = x_entry
    #     # print(f"True Action Traj:{action_traj}, True Reward Traj:{reward_vec}")
    #     rewards_sum = np.zeros(5)
    #     print(f"Reward_traj for inferrence not normalized:{reward_vec}")
    #     # print(min_max_normalize(min_v=min_value, max_v=max_value, target=rewards_sum))
    #     reward_vec = min_max_normalize(min_v=min_value, max_v=max_value, target=reward_vec)
    #     pref = np.round(inferrer.inferring_model(np.array([reward_vec])).numpy(), 2)
    #     # KL = scipy.stats.entropy(pref[0], sum_normalize(ground_truth_dict[key])+1e-5)
    #     # print(f"Inferred:{pref}, Ground Truth:{sum_normalize(ground_truth_dict[key])}, KL:{KL}")
    #     print(f"Inferred:{pref}")
    #     print(" =================================================================== \n")
    run_time = datetime.now() - start_time
    print(f'Inferrence Time DST: {run_time} s')
    for key in ground_truth_dict.keys():
        traffic_env.reset()
        reward_vec, weights, action_traj = dqn_ag.play_episode(dqn_ag.normalize_pref(ground_truth_dict[key]))
        reward_vec[0] += 2
        print(f"True Action Traj:{action_traj}, True Reward Traj:{reward_vec}")
        rewards_sum = np.zeros(5)

        # print(min_max_normalize(min_v=min_value, max_v=max_value, target=rewards_sum))
        # reward_vec = min_max_normalize(min_v=min_value, max_v=max_value, target=reward_vec)

        pref = np.round(inferrer.inferring_model(np.array([reward_vec])).numpy(), 2)
        reward_vec, weights, action_traj = dqn_ag.play_episode(pref[0])
        print(f"Reward_traj for inferrence not normalized:{reward_vec}")
        KL = scipy.stats.entropy(pref[0], sum_normalize(ground_truth_dict[key]) + 1e-5)
        print(f"Inferred:{pref}, Ground Truth:{sum_normalize(ground_truth_dict[key])}, KL:{KL}")
        print(" =================================================================== \n")
    run_time = datetime.now() - start_time
    print(f'Inferrence Time DST: {run_time} s')
