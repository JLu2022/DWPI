from mo_traffic_agent import PreferenceSpace, DQNAgent
from mo_traffic_env import MOTraffic
import keras
import numpy as np
import random

mo_traffic_env = MOTraffic()
image = mo_traffic_env.reset()
pref_space = PreferenceSpace()
model_path = "C://Users//19233436//PycharmProjects//DWPI//Environments//Traffic//Traffic_Model//200000"
save_trajs_to = "C://Users//19233436//PycharmProjects//DWPI//Trajectories//MOTraffic//"
dqn_ag = DQNAgent(mo_traffic_env, model_path=None)
dqn_ag.model = keras.models.load_model(model_path)
TEMPERATURE = 2
preferences = [np.array([1, 50, 10, 50, 1]), np.array([10, 50, 10, 10, 1]),
               np.array([5, 50, 0, 50, 1]), np.array([1, 50, 0, 50, 1])]
# "Always_Safe" // "Always_Fast" // "Fast_Safe" // "Slow_Safe"

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
        # normalized_pref = dqn_ag.normalize_pref(pref)
        reward_vec, _, states_traj = dqn_ag.generate_experience(episode=0, epsilon=0, weight_i=pref,
                                                                scene="train",
                                                                deterministic_ratio=deterministic_ratio,
                                                                num_trajs=num_of_trajs, temperature=TEMPERATURE)
        print(f"Preference:{pref}, AVG Reward_vec:{np.mean(reward_vec, axis=0)}\nstate traj:{np.mean(states_traj, axis=0)}")
        print("------------------------------------------------------------------------------")
        for _ in range(50):
            preference_labels.append(np.array(pref))
            vec_r_trajs.append(np.mean(reward_vec, axis=0))
            states_trajs.append(np.mean(states_traj, axis=0))

    np.save(save_trajs_to + "deter_ratio_" + str(deterministic_ratio) + "_" + "//vec_r_trajs.npy",
            vec_r_trajs)
    np.save(save_trajs_to + "deter_ratio_" + str(deterministic_ratio) + "_" + "//states_trajs.npy",
            states_trajs)
    np.save(save_trajs_to + "deter_ratio_" + str(deterministic_ratio) + "_" + "//labels.npy",
            preference_labels)
    # 0:step, 1:treasure, 2:not walk on road 3: collision 4: on the wall
