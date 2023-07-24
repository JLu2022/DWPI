import numpy as np
from item_gathering_agent import PreferenceSpace, DQNAgent
from item_gathering_env import ItemGathering
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras
import keras
from keras import layers, backend as K
from keras.losses import mse, CosineSimilarity, cosine_similarity, mean_absolute_error
from datetime import datetime  # Used for timing script
from sklearn.metrics import mean_squared_error
import scipy.stats

SEED = 42
NUM_WEIGHTS = 4


class Inferrer(object):
    def __init__(self, actor):
        self.inferring_model = self.build_model()
        self.actor = actor

    def build_model(self):
        input_dim = 6
        output_dim = 4

        original_inputs = tf.keras.Input(shape=(input_dim,), name="encoder_input")
        x = layers.Dense(64, activation="relu")(original_inputs)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        z = layers.Dense(output_dim, name="z")(x)

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


def normalize(min_v, max_v, target):
    return (target - min_v) / (max_v - min_v)


def new_normalize(input_list):
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
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    # For timing the script
    start_time = datetime.now()
    item_env = ItemGathering()

    # Instantiate agent (pass in environment)
    agent_model_path = "C://Users//19233436//PycharmProjects//DWPI//Environments//ItemGathering//ItemGatheringModel//200000"
    x_reward_path = "C://Users//19233436//PycharmProjects//DWPI//Environments//ItemGathering//x_train_raw_fixed.npy"
    y_reward_path = "C://Users//19233436//PycharmProjects//DWPI//Environments//ItemGathering//y_train_raw_fixed.npy"
    infer_model_path = "C://Users//19233436//PycharmProjects//DWPI//Environments//ItemGathering//InferringModel//"
    dqn_ag = DQNAgent(env=item_env, model_path=agent_model_path)
    dqn_ag.model = keras.models.load_model(agent_model_path)
    inferer = Inferrer(actor=dqn_ag)

    raw_x_reward = np.load(x_reward_path)
    raw_y_train_pref = np.load(y_reward_path)
    # print(f"raw_x_reward:{raw_x_reward}\traw_y_pref:{raw_y_train_pref}")
    x_train = raw_x_reward
    y_train = np.array([y_entry[2:] for y_entry in raw_y_train_pref])
    # print(f"x_train:{x_train}\n"
    #       f"y_train:{y_train}")
    # inferer.train(x=x_train, y=y_train, batch_size=64, epochs=2000)
    # keras.models.save_model(inferer.inferring_model, filepath=infer_model_path)
    infer_model = keras.models.load_model(infer_model_path)
    w_dictionary = {
        'Competitive': np.array([-1, -5, +10, +20, +10, -20]),  # Competitive
        'Cooperative': np.array([-1, -5, +10, +20, +10, +20]),  # Cooperative
        'Fair': np.array([-1, -5, +20, +15, +20, +20]),  # Fair
        'Generous': np.array([-1, -5, +20, 0, +20, +20]),  # Generous
    }

    num_of_trajs = 100
    ALE = 5
    EPISODES = 8000
    preference_upper_bound = 20
    preference_lower_bound = -20
    # DWPI
    print("\n\n DWPI start>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    deterministic_ratio = 0.8
    for key in w_dictionary.keys():
        w_true = w_dictionary[key]  # get ground truth
        cooperative = 0 if w_true[-1] < 0 else 1
        normalized_w = abs(w_true) / sum(w_true)

        # print(f"normalized_w:{normalized_w}")
        sum_reward_vec = np.zeros(6)
        for _ in range(num_of_trajs):
            reward_vec, weights, _ = dqn_ag.play_episode(normalized_w, cooperate=cooperative,
                                                         deterministic_ratio=deterministic_ratio)
            sum_reward_vec += reward_vec

        avg_reward_true = sum_reward_vec / num_of_trajs
        w_infer = infer_model(np.array([avg_reward_true]))
        w_infer = tf.squeeze(w_infer).numpy()
        # print(f"w_infer:{w_infer}")
        w_infer = np.maximum(w_infer, -20)
        w_infer = np.minimum(w_infer, 20)
        for i in range(len(w_infer[:-1])):
            if w_infer[i] < 0:
                w_infer[i] = 0
        w_infer = np.insert(w_infer, 0, -5)
        w_infer = np.insert(w_infer, 0, -1)

        normalized_w_infer = abs(w_infer) / sum(w_infer)
        # show_inference = abs(w_infer) / sum(abs(w_infer))
        sum_reward_vec = np.zeros(6)
        for _ in range(num_of_trajs):
            reward_vec, weights, _ = dqn_ag.play_episode(normalized_w, cooperate=cooperative,
                                                         deterministic_ratio=1)
            sum_reward_vec += reward_vec

        avg_reward_true = sum_reward_vec / num_of_trajs
        sum_reward_vec = np.zeros(6)
        for _ in range(num_of_trajs):
            inferred_reward, weights, _ = dqn_ag.play_episode(normalized_w_infer, cooperate=cooperative,
                                                              deterministic_ratio=1)
            sum_reward_vec += inferred_reward
        avg_reward_infer = sum_reward_vec / num_of_trajs
        near_zero = 1e-5

        print(
            f"Deterministic Ratio:{deterministic_ratio}\n"
            f"True Weight:{w_true}\n"
            f"Infer Weight:{np.round(w_infer, 0)}\n"
            f"True reward:{np.round(avg_reward_true, 2)}\n"
            f"Infer reward:{np.round(avg_reward_infer, 2)}\n"
        )
        print(
            "\n======================================================================================================")

    run_time = datetime.now() - start_time
    print(f'DWPI model Inferring Run time: {run_time} s')
    print("\n\n PM start>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    start_time = datetime.now()

    # PM tryout
    # for key in ground_truth_dict.keys():
    #     for sub_optimal in [False, True]:
    #         if sub_optimal:
    #             sub_noise = random.randint(2, 3)
    #         else:
    #             sub_noise = 0
    #         preference_space = PreferenceSpace()
    #         ground_truth_preference = ground_truth_dict[key]
    #         ground_truth_cooperative = (ground_truth_dict[key][-1] > 0)
    #         normalized_ground_truth = abs(ground_truth_preference) / sum(abs(ground_truth_preference))
    #         sum_ground_reward = np.zeros(6)
    #         for _ in range(try_outs):
    #             ground_reward, _, _ = dqn_ag.play_episode(normalized_ground_truth, cooperate=ground_truth_cooperative)
    #             sum_ground_reward += ground_reward
    #         avg_ground_reward = sum_ground_reward / try_outs
    #         ground_truth_utility = np.dot(avg_ground_reward, normalized_ground_truth)
    #         avg_ground_reward[0] += sub_noise
    #         for _ in range(ALE):
    #             inferred_preference, cooperative = preference_space.sample()
    #             item_env = ItemGathering()
    #             pm_dqn = DQNAgent(env=item_env, model_path=None)
    #             pm_dqn.train_model(episodes=EPISODES, preference=inferred_preference, cooperate=cooperative)
    #
    #             abs_preference = abs(inferred_preference)  # make non-negative preference vector
    #             sum_pref = sum(abs_preference)
    #             normalized_pref = abs_preference / sum_pref
    #             PM_reward_vec, _, _ = pm_dqn.play_episode(normalized_pref, cooperate=cooperative)
    #             # if mse(PM_reward_vec, avg_ground_reward) < 10:
    #             #     break
    #
    #         for _ in range(try_outs):
    #             ground_reward, _, _ = dqn_ag.play_episode(normalized_pref, cooperate=cooperative)
    #             sum_ground_reward += ground_reward
    #         avg_inferred_reward = sum_ground_reward / try_outs
    #         inferred_utility = np.dot(avg_inferred_reward, normalized_pref)
    #         near_zero = 1e-5
    #         print(f"Sub-optimal?:{sub_optimal}\n"
    #               f"Gound_truth:{normalized_ground_truth}\n"
    #               f"inferred:{normalized_pref}\n"
    #               f"KL:{scipy.stats.entropy(normalized_ground_truth + near_zero, normalized_pref + near_zero)}\n"
    #               f"MSE:{mean_squared_error(normalized_ground_truth, normalized_pref)}\n"
    #               f"Absolute Utility Error{abs(ground_truth_utility - inferred_utility)}")
    #         print(
    #             "=========================================================================================================")
    # run_time = datetime.now() - start_time
    # print(f'PM model Inferring Run time: {run_time} s')
    #
    #
    #
    # print("\n\n MWAL start>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # # MWAL tryout
    # start_time = datetime.now()
    # for key in ground_truth_dict.keys():
    #     for sub_optimal in [False, True]:
    #         if sub_optimal:
    #             sub_noise = random.randint(2, 3)
    #         else:
    #             sub_noise = 0
    #         preference_space = PreferenceSpace()
    #         ground_truth_preference = ground_truth_dict[key]
    #         ground_truth_cooperative = (ground_truth_dict[key][-1] > 0)
    #         normalized_ground_truth = abs(ground_truth_preference) / sum(abs(ground_truth_preference))
    #         sum_ground_reward = np.zeros(6)
    #         for _ in range(try_outs):
    #             ground_reward, _, _ = dqn_ag.play_episode(normalized_ground_truth,
    #                                                       cooperate=ground_truth_cooperative)
    #             sum_ground_reward += ground_reward
    #         avg_ground_reward = sum_ground_reward / try_outs
    #         ground_truth_utility = np.dot(avg_ground_reward, normalized_ground_truth)
    #         avg_ground_reward[0] += sub_noise
    #         inferred_preference, cooperative = preference_space.sample()
    #         for _ in range(ALE):
    #             item_env = ItemGathering()
    #             pm_dqn = DQNAgent(env=item_env, model_path=None)
    #             pm_dqn.train_model(episodes=EPISODES, preference=inferred_preference, cooperate=cooperative)
    #
    #             abs_preference = abs(inferred_preference)  # make non-negative preference vector
    #             sum_pref = sum(abs_preference)
    #             normalized_pref = abs_preference / sum_pref
    #             reward_vec, _, _ = pm_dqn.play_episode(normalized_pref, cooperate=cooperative)
    #             beta = (1.0 + np.sqrt(2 * np.log(6) / ALE)) ** (-1.0)
    #             W = np.zeros(6)
    #             for i in range(len(normalized_pref)):
    #                 W[i] = normalized_pref[i] * beta ** (reward_vec[i] - avg_ground_reward[i])
    #             for j in range(len(normalized_pref)):
    #                 normalized_pref[j] = W[j] / sum(W)
    #             # if mse(reward_vec, avg_ground_reward) < 10:
    #             #     break
    #
    #         for _ in range(try_outs):
    #             ground_reward, _, _ = dqn_ag.play_episode(normalized_pref, cooperate=cooperative)
    #             sum_ground_reward += ground_reward
    #         avg_inferred_reward = sum_ground_reward / try_outs
    #         inferred_utility = np.dot(avg_inferred_reward, normalized_pref)
    #         near_zero = 1e-5
    #         print(f"Sub-optimal?:{sub_optimal}\n"
    #               f"Gound_truth:{normalized_ground_truth}\n"
    #               f"inferred:{normalized_pref}\n"
    #               f"KL:{scipy.stats.entropy(normalized_ground_truth + near_zero, normalized_pref + near_zero)}\n"
    #               f"MSE:{mean_squared_error(normalized_ground_truth, normalized_pref)}\n"
    #               f"Absolute Utility Error{abs(ground_truth_utility - inferred_utility)}")
    #         print(
    #             "=========================================================================================================")
    # run_time = datetime.now() - start_time
    # print(f'MWAL model Inferring Run time: {run_time} s')
