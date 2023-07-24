from item_gathering_agent import PreferenceSpace, DQNAgent
from item_gathering_env import ItemGathering
import keras
import numpy as np
import random

item_gathering_env = ItemGathering()
image = item_gathering_env.reset()
pref_space = PreferenceSpace()
model_path = "C://Users//19233436//PycharmProjects//DWPI//Environments//ItemGathering//ItemGatheringModel200000"
save_trajs_to = "C://Users//19233436//PycharmProjects//DWPI//Trajectories//item_gathering//"
dqn_ag = DQNAgent(item_gathering_env, model_path=None)
dqn_ag.model = keras.models.load_model(model_path)
TEMPERATURE = 2
# preferences = []
p_step = 1  # Time Penalty
p_wall = 5  # Wall
# for p_green in range(0, 21, 5):
#     for p_red in range(0, 21, 5):
#         for p_yellow in range(0, 21, 5):
#             for p_other_red in range(-20, 21, 5):
#                 preference = np.array([p_step, p_wall, p_green, p_red, p_yellow, abs(p_other_red), p_other_red > 0], dtype=np.float32)
#                 preferences.append(preference)
preferences = [np.array([1, 5, 10, 20, 10, -20]), np.array([1, 5, 10, 20, 10, 20]),
               np.array([1, 5, 20, 15, 20, 20]), np.array([1, 5, 20, 0, 20, 20])]
"""
Reward: R_step, R_wall, R_green, R_red, R_yellow, R_other_red
"""
x_train_raw = []
y_train_raw = []
num_of_trajs = 100
data_augmented_num = 50
deterministic_ratios = [0.5]
for deterministic_ratio in deterministic_ratios:
    print(f"Generating traj with deterministic_ratio:{deterministic_ratio}")
    vec_r_trajs = []
    scalar_r_trajs = []
    states_trajs = []
    states_actions_trajs = []
    preference_labels = []
    for pref in preferences:
        reward_vec, _, states_traj = dqn_ag.generate_experience(episode=0, epsilon=0, weight_i=pref, scene="train",
                                                   deterministic_ratio=deterministic_ratio,
                                                   num_trajs=num_of_trajs, temperature=TEMPERATURE)
        print(f"Preference:{pref}, AVG Reward_vec:{np.mean(reward_vec, axis=0)}")
        print("------------------------------------------------------------------------------")
        for _ in range(50):
            preference_labels.append(np.array(pref[2:]))
            vec_r_trajs.append(np.mean(reward_vec, axis=0))
            states_trajs.append(np.mean(states_traj, axis=0))


    np.save(save_trajs_to + "deter_ratio_" + str(deterministic_ratio) + "_" + "//vec_r_trajs.npy",
            vec_r_trajs)
    np.save(save_trajs_to + "deter_ratio_" + str(deterministic_ratio) + "_" + "//labels.npy",
            preference_labels)
    np.save(save_trajs_to + "deter_ratio_" + str(deterministic_ratio) + "_" + "//states_trajs.npy",
            states_trajs)
