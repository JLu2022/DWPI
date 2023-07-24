import numpy as np
import random
from datetime import datetime  # Used for timing script
from sklearn.metrics import mean_squared_error
# from Environments.ItemGathering.item_gathering_agent import PreferenceSpace, DQNAgent
from Environments.Traffic.mo_traffic_agent import PreferenceSpace as Traffic_P_Space, DQNAgent as traffic_agent
from Environments.ItemGathering.item_gathering_agent import PreferenceSpace as Item_Gathering_P_Space, \
    DQNAgent as item_gathering_agent
from Agents.Q_Agent.tabular_q_agent import Tabular_Q_Agent
from Utils.utils import filter_pref
from sklearn.preprocessing import MinMaxScaler


def random_sample_pref():
    p0 = random.choice([x for x in range(1, 101)])  # Green
    p1 = 1 - p0 / 100  # get treasure
    preference_infer = np.array([p0 / 100, p1], dtype=np.float32)
    return preference_infer


scaler = MinMaxScaler()


class MWALPreferenceSpace:
    def __init__(self, weight):
        self.weight = weight

    def sample(self):
        return self.weight


def get_mu(env, action_trajectory, gamma=1, mu_dimension=5):
    env.reset()
    reward_trajectory = np.zeros(mu_dimension)
    for action in action_trajectory:
        rewards, state, done, _ = env.step(action)
        reward_trajectory = rewards + gamma * reward_trajectory
    return reward_trajectory


def random_sample_pref():
    p0 = random.choice([x for x in range(1, 101)])  # Green
    p1 = 1 - p0 / 100  # get treasure
    preference_infer = np.array([p0 / 100, p1], dtype=np.float32)
    return preference_infer


expert_trajectory = []


def project_method(expert, true_pref, num_of_traj, env, episodes, num_of_iterations, mu_dimension=5, temperature=2,
                   deterministic_ratio=0.5, env_name="Traffic", truncated_thres=2):
    if env_name == "Traffic":
        preference_space = Traffic_P_Space()
    if env_name == "Item Gathering":
        preference_space = Item_Gathering_P_Space()

    mu_expert = np.zeros(mu_dimension)
    for _ in range(num_of_traj):
        _, expert_action_traj, _ = expert.generate_experience(episode=0, epsilon=0, weight_i=true_pref,
                                                              scene="train", deterministic_ratio=deterministic_ratio,
                                                              num_trajs=1, temperature=temperature,
                                                              truncated_thres=truncated_thres)
        mu_expert += get_mu(env=env, action_trajectory=expert_action_traj, mu_dimension=mu_dimension)

    mu_expert = mu_expert / num_of_traj
    start_time = datetime.now()
    min_loss = np.inf
    pref_loss_dict = {}
    for turn in range(1, num_of_iterations + 1):
        if env_name == "DST":
            preference = random_sample_pref()
        else:
            preference = preference_space.sample()
        if env_name == "Traffic":
            agent = traffic_agent(env=env, model_path=None)
        if env_name == "Item Gathering":
            agent = item_gathering_agent(env=env, model_path=None)
        if env_name == "DST":
            agent = Tabular_Q_Agent(env=env)

        if not env_name == "DST":
            agent.train_model(episodes, pref_space=None, preference=preference,
                              save_per=500000,
                              show_detail_per=1000000)
        else:
            agent.q_learning(episodes=episodes)
        mu = np.zeros(mu_dimension)
        for i in range(num_of_traj):
            _, action_list, _ = agent.generate_experience(
                episode=0, epsilon=0, weight_i=preference, scene="train",
                deterministic_ratio=1, num_trajs=1, temperature=temperature, truncated_thres=truncated_thres)

            mu += get_mu(env=env, action_trajectory=action_list, mu_dimension=mu_dimension)
        mu = mu / num_of_traj
        loss = mean_squared_error(mu_expert, mu)
        if loss < min_loss:
            min_loss = loss
        pref_loss_dict[loss] = preference
    run_time = datetime.now() - start_time
    return run_time, pref_loss_dict[min_loss]


def MWAL(expert, true_pref, num_of_traj, env, episodes, AL_ite, mu_dimension, temperature, deterministic_ratio,
         start_pref, Reward_Mask_Max, Reward_Mask_Min, env_name, truncated_thres):
    mu_expert = np.zeros(mu_dimension)
    for _ in range(num_of_traj):
        _, expert_action_traj, _ = expert.generate_experience(episode=0, epsilon=0, weight_i=true_pref,
                                                              scene="train", deterministic_ratio=deterministic_ratio,
                                                              num_trajs=1,
                                                              temperature=temperature, truncated_thres=truncated_thres)
        mu_expert += get_mu(env=env, action_trajectory=expert_action_traj, mu_dimension=mu_dimension)
    mu_expert = mu_expert / num_of_traj

    start_time = datetime.now()
    w_infer = np.array(start_pref)
    for turn in range(1, AL_ite + 1):
        if env_name == "Traffic":
            agent = traffic_agent(env=env, model_path=None)
        if env_name == "Item Gathering":
            agent = item_gathering_agent(env=env, model_path=None)
        else:
            agent = Tabular_Q_Agent(env=env)

        if not env_name == "DST":
            agent.train_model(episodes, pref_space=None, preference=tuple(w_infer),
                              save_per=500000,
                              show_detail_per=1000000)
        else:
            agent.q_learning(episodes=episodes)
        mu = np.zeros(mu_dimension)
        for _ in range(num_of_traj):
            _, action_list, _ = agent.generate_experience(
                episode=0, epsilon=0, weight_i=w_infer, scene="train",
                deterministic_ratio=1, num_trajs=1, temperature=2, truncated_thres=truncated_thres)
            mu += get_mu(env=env, action_trajectory=action_list, mu_dimension=mu_dimension)
        mu /= num_of_traj

        beta = (1.0 + np.sqrt(2 * np.log(mu_dimension) / AL_ite)) ** (-1.0)
        w_infer = scaler.fit_transform(w_infer.reshape(-1, 1))
        for i in range(len(w_infer) - 1):
            w_infer[i] = w_infer[i] * beta ** (mu[i] - mu_expert[i])
        w_infer = scaler.inverse_transform(w_infer)
        w_infer = filter_pref(w_infer, max_p=Reward_Mask_Max, min_p=Reward_Mask_Min).ravel()
    run_time = datetime.now() - start_time

    return run_time, w_infer
