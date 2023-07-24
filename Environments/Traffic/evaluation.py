from mo_traffic_agent import MOTraffic, PreferenceSpace, DQNAgent
import keras
import numpy as np
import random
traffic_env = MOTraffic()
image = traffic_env.reset()
pref_space = PreferenceSpace()
dqn_ag = DQNAgent(traffic_env,model_path=None)
dqn_ag.model = keras.models.load_model("C://Users//Junlin//Inverse_MORL//Experiment//Traffic//Traffic_Model//200000")
dqn_ag.target_model = keras.models.load_model("C://Users//Junlin//Inverse_MORL//Experiment//Traffic//Traffic_Model//200000")
preferences = []
for p_0 in range(1,11):
    for p_2 in range(0,21,10):
        for p_3 in range(0,51,10):
            preference = [p_0,50,p_2,p_3,1]
            preferences.append(np.array(preference))
preferences = [np.array([1, 50, 10, 50, 1]),np.array([10, 50, 10, 10, 1]),np.array([5, 50, 0, 50, 1]),np.array([1, 50, 0, 50, 1])]
x_train_raw = []
y_train_raw = []
try_outs = 20
for pref in preferences:
    sum_reward_vec = np.zeros(5)
    weights = np.zeros(3)
    sum_utility = 0
    sum_inferred_utility = 0
    for _ in range(try_outs):
        p = dqn_ag.normalize_pref(pref)
        ground_truth_p = np.array([0.01, 0.49, 0.00, 0.49, 0.01])
        reward_vec, weights,action_traj = dqn_ag.play_episode(preference=ground_truth_p)
        utility_true = np.dot(ground_truth_p,reward_vec)

        inferred_p = np.array([0.99, 0.01, 0.00, 0.00, 0.00])
        reward_vec, weights, action_traj = dqn_ag.play_episode(preference=inferred_p)
        utility_inference = np.dot(inferred_p, reward_vec)
        # suboptimal_steps = random.randint(0,3)
        # noisy_reward_vec = np.array([reward_vec[0]-suboptimal_steps,reward_vec[1], reward_vec[2], reward_vec[3],reward_vec[4]])
        sum_reward_vec+=reward_vec
        sum_utility += utility_true
        sum_inferred_utility += utility_inference
        # x_train_raw.append(noisy_reward_vec)
        # y_train_raw.append(pref)
    print("Preference:{}, AVG Reward_vec:{},weights:{}".format(pref,sum_reward_vec/try_outs, weights))
    print("avg_true_utility:{}, avg_infer_utility:{},absolute error:{}".format(sum_utility/try_outs,sum_inferred_utility/try_outs,(sum_utility/try_outs)-(sum_inferred_utility/try_outs)))
# x_train_raw = np.array(x_train_raw)
# y_train_raw = np.array(y_train_raw)
# np.save("Dataset//x_train_raw.npy",x_train_raw)
# np.save("Dataset//y_train_raw.npy",y_train_raw)