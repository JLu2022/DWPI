from Agents.Q_Agent.tabular_q_agent import Tabular_Q_Agent
import numpy as np
import random
from datetime import datetime  # Used for timing script
from sklearn.metrics import mean_squared_error
from Environments.ItemGathering.item_gathering_agent import PreferenceSpace

class MWALPreferenceSpace:
    def __init__(self, weight):
        self.weight = weight

    def sample(self):
        return self.weight


def get_mu(env, action_trajectory, gamma=1):
    env.reset()
    reward_trajectory = np.zeros(2)
    for action in action_trajectory:
        state, rewards, done = env.step(action)
        # print(f"rewards:{rewards}")
        reward_trajectory = rewards + gamma * reward_trajectory
    return reward_trajectory


def random_sample_pref():
    p0 = random.choice([x for x in range(1, 101)])  # Green
    p1 = 1 - p0 / 100  # get treasure
    preference_infer = np.array([p0 / 100, p1], dtype=np.float32)
    return preference_infer



expert_trajectory = []


def imorl(expert, true_pref, num_of_traj, env, episodes, num_of_iterations):
    mu_expert = np.zeros(2)
    for _ in range(num_of_traj):
        _, expert_action_traj, _ = expert.generate_experience(episode=0, epsilon=0, weight_i=tuple(true_pref),
                                                              scene="train", deterministic_ratio=0.5, num_trajs=1,
                                                              temperature=2, reward_mask=True)
        mu_expert += get_mu(env=env, action_trajectory=expert_action_traj)

    mu_expert = mu_expert / num_of_traj
    print(f"avg_mu_expert:{mu_expert}")
    start_time = datetime.now()
    w_infer = [0, 1]
    min_loss = np.inf
    for turn in range(1, num_of_iterations + 1):
        preference = random_sample_pref()
        agent = Tabular_Q_Agent(env=env)
        agent.preference_list = [tuple(preference)]
        agent.q_learning(episodes=episodes)

        _, _, _, _, action_list = agent.generate_experience(
            episode=0, epsilon=0, weight_i=tuple(preference), scene="train",
            deterministic_ratio=1, num_trajs=1, temperature=2, reward_mask=True)

        mu = get_mu(env=env, action_trajectory=action_list)
        if mean_squared_error(mu_expert, mu) < min_loss:
            min_loss = mean_squared_error(mu_expert, mu)
            w_infer = preference
        if mean_squared_error(mu_expert, mu) < 0.5:
            # print(f"w_infer:{w_infer}\tw_true:{true_pref}")
            break
    run_time = datetime.now() - start_time
    return run_time, tuple(w_infer)


def MWAL(expert, true_pref, num_of_traj, env, episodes, AL_ite):
    mu_expert = np.zeros(2)
    for _ in range(num_of_traj):
        _, _, _, _, expert_action_traj = expert.generate_experience(episode=0, epsilon=0, weight_i=tuple(true_pref),
                                                                    scene="train", deterministic_ratio=0.5, num_trajs=1,
                                                                    temperature=2, reward_mask=True)
        mu_expert += get_mu(env=env, action_trajectory=expert_action_traj)
    mu_expert = mu_expert / num_of_traj

    start_time = datetime.now()
    w_infer = np.array([0.5, 0.5])
    min_loss = np.inf
    for turn in range(1, AL_ite + 1):
        # preference = random_sample_pref()
        agent = Tabular_Q_Agent(env=env)
        agent.preference_list = [tuple(w_infer)]
        agent.q_learning(episodes=episodes)
        _, _, _, _, action_list = agent.generate_experience(
            episode=0, epsilon=0, weight_i=tuple(w_infer), scene="train",
            deterministic_ratio=1, num_trajs=1, temperature=2, reward_mask=True)
        mu = get_mu(env=env, action_trajectory=action_list)
        mse_mu_mu_expert = mean_squared_error(mu_expert, mu)
        if mse_mu_mu_expert < min_loss:
            min_loss = mse_mu_mu_expert
        # if mean_squared_error(mu_expert, mu) <= 2:
        #     break
        beta = (1.0 + np.sqrt(2 * np.log(2) / AL_ite)) ** (-1.0)
        for i in range(len(w_infer)):
            w_infer[i] = w_infer[i] * beta ** (mu[i]-mu_expert[i])
        w_infer = w_infer / sum(w_infer)
        # print(f"w_infer:{w_infer}\tw_true:{true_pref}\nmin_loss:{min_loss}\nmu:{mu}\tmu_expert:{mu_expert}")
    run_time = datetime.now() - start_time

    return run_time, tuple(w_infer)
