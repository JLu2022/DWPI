import csv
import random
# import line_profiler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Environments.DeepSeaTreasure.DiscreteDST import DeepSeaTreasureEnvironment
from scipy.special import softmax
import copy

ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right"}
Training_visualization_path = "C://Users//19233436//PycharmProjects//DWPI//Train_Agent//Train Result//discreteDST_Result//Learning_curve//"


class MOTabularAgent:
    def __init__(self, env):
        self.env = env
        self.actions = [i for i in range(len(env.actions))]
        self.preference_list = [(np.round(w0 / 100, 2), np.round(1 - w0 / 100, 2)) for w0 in range(0, 101)]

        self.Q = np.random.rand(self.env.grid_rows, self.env.grid_cols, len(self.env.actions))

        self.time_Q = np.random.rand(self.env.grid_rows, self.env.grid_cols, len(self.env.actions))
        self.target_time_Q = copy.deepcopy(self.time_Q)
        # self.time_Q = np.zeros((self.env.grid_rows, self.env.grid_cols, len(self.env.actions)))
        self.treasure_Q = np.random.rand(self.env.grid_rows, self.env.grid_cols, len(self.env.actions))
        self.target_treasure_Q = copy.deepcopy(self.treasure_Q)
        # self.treasure_Q = np.zeros((self.env.grid_rows, self.env.grid_cols, len(self.env.actions)))
        for forbidden_state in self.env.forbidden_states:
            self.time_Q[forbidden_state] = np.full(len(self.env.actions), -200)

        for treasure_location in self.env.treasure_locations:
            self.treasure_Q[treasure_location] = np.zeros(len(self.env.actions))
        # print(self.Q)
        # print(self.time_Q)
        # print(self.treasure_Q)

    def envelope_epsilon_greedy(self, state, epsillon, weight):
        if np.random.rand() < epsillon:
            return np.random.choice(self.actions), None, None, None
        else:
            Q_values = np.sum(weight * np.array([self.time_Q[state], self.treasure_Q[state]]).T, axis=1)
            max_Q = np.max(Q_values)
            indices = np.where(Q_values == max_Q)
            action = random.choice(indices[0])
            return action, Q_values, self.time_Q[state], self.treasure_Q[state]

    def epsilon_greedy_policy(self, state, epsillon):
        if np.random.rand() < epsillon:
            return np.random.choice(self.actions), None, None, None
        else:
            action = np.argmax(self.Q[state])
            return action, self.Q[state], None, None

    def soft_max_q(self, state, weight, temperature=1.4):
        Q_values = np.sum(weight * np.array([self.time_Q[state], self.treasure_Q[state]]).T, axis=1)
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

    def scalarise(self, rewards, weights):
        rewards = np.array(rewards)
        return np.dot(rewards, weights)

    def initialise_q_values(self):
        self.Q_dict = {}
        for weights in self.preference_list:
            Q_values = np.random.rand(self.env.grid_rows, self.env.grid_cols, len(self.env.actions)) * 100
            self.Q_dict[weights] = Q_values

        for key in self.Q_dict:
            for forbidden_state in self.env.forbidden_states:
                self.Q_dict[key][forbidden_state] = np.full(len(self.env.actions), -200)

            for treasure_location in self.env.treasure_locations:
                self.Q_dict[key][treasure_location] = np.zeros(len(self.env.actions))
        return self.Q_dict

    def plot_learning_curve(self, stats, key, weights):
        """
        Plot the rewards per episode collected during training
        """
        fig, ax = plt.subplots()
        ax.plot(stats[:, 0], stats[:, 1])
        ax.set_xlabel('episode')
        ax.set_ylabel('reward per episode')
        ax.set_title(f'time, treasure weighting: {key}')
        plt.savefig(Training_visualization_path + str(weights) + ".png")
        plt.close()

    def conventional_train(self, episodes):
        print("**********************CONVENTIONAL TRAINING*******************************")
        epsillon0 = 0.8
        alpha = 0.05
        gamma = 1

        for weights in self.preference_list:
            for i in range(1, episodes + 1):
                r_TDs = []
                actions = []
                state = self.env.reset()

                epsillon = max(epsillon0 - i / episodes, 0.01)

                while True:
                    action, _, _, _ = self.epsilon_greedy_policy(state, epsillon)
                    # print(action)
                    next_state, rewards, done = self.env.step(action)
                    reward = self.scalarise(rewards, weights)
                    r_TD = reward + gamma * np.max(self.Q[next_state]) - self.Q[(*state, action)]
                    self.Q[(*state, action)] += alpha * r_TD
                    r_TDs.append(r_TD)
                    actions.append(ACTIONS[action])
                    if done:
                        break
                    state = next_state
                if i % 10000 == 0:
                    print(f"Episode:{i}--> Epsilon:{epsillon}\n"
                          f"Preference:{weights}\n"
                          f"R TD:{np.mean(np.array(r_TDs))}\n"
                          f"Action traj:{actions}\n"
                          f"Rewards:{rewards}")
                    print("==================================================")

    def q_learning(self, num_train_rounds, epochs=20):
        epsillon0 = 0.8
        alpha = 0.01
        gamma = 1
        beta = 0.2
        # self.stats_dict = {}
        # self.initialise_q_values()
        for j in range(0, epochs):
            weights = random.choice(self.preference_list)
            # weights = self.preference_list[j]
            for i in range(1, num_train_rounds + 1):
                state = self.env.reset()
                epsillon = max(epsillon0 - i / num_train_rounds, 0.1)
                # epsillon = 0.6
                time_TDs = []
                treasure_TDs = []
                r_TDs = []
                actions = []
                while True:
                    action, Q_values, time_Q, treasure_Q = self.envelope_epsilon_greedy(state, epsillon, weights)
                    # print(f"Q_values:{Q_values}\tq_time:{time_Q}\tq_treasure:{treasure_Q}")
                    next_state, rewards, done = self.env.step(action)

                    r_time = rewards[0]
                    r_treasure = rewards[1]
                    r = np.sum(weights * np.array([r_time, r_treasure]).T)

                    # print(f"reward:{r}\ttime_r:{r_time}\ttreasure_r:{r_treasure}")
                    # if Q_values is not None and state == (0,0):
                    #     print(f"state:{state}\nQ:{Q_values}\nQ_time:{time_Q}\nQ_treasure:{treasure_Q}\nReward:{rewards}")
                    #     print("--------------------------------------------------------")
                    next_time_Q = self.target_time_Q[next_state]
                    next_treasure_Q = self.target_treasure_Q[next_state]
                    next_Q_values = np.sum(weights * np.array([self.time_Q[next_state], self.treasure_Q[next_state]]).T,
                                           axis=1)

                    max_index = np.argmax(next_Q_values)
                    print(f"Q-time:{next_time_Q}")
                    # print(f"next_time:{next_time_Q}\t next_treasure:{next_treasure_Q}\t next_q:{next_Q_values}")
                    time_TD = r_time + (1 - done) * (gamma * next_time_Q[max_index]) - self.time_Q[(*state, action)]
                    treasure_TD = r_treasure + (1 - done) * (gamma * next_treasure_Q[max_index]) - self.treasure_Q[
                        (*state, action)]
                    r_TD = r + (1 - done) * (gamma * np.max(next_Q_values)) - np.sum(
                        weights * np.array([self.time_Q[(*state, action)], self.treasure_Q[(*state, action)]]).T)

                    # print(f"reward:{r}\t next_Q:{np.max(next_Q_values)}\t r_TD:{r_TD}")

                    loss_A_factor = (num_train_rounds - i) / num_train_rounds
                    loss_A_factor = 0.8
                    loss_B_factor = i / num_train_rounds
                    loss_B_factor = 0.2

                    self.time_Q[(*state, action)] += alpha * (loss_A_factor * time_TD + loss_B_factor * r_TD)
                    self.treasure_Q[(*state, action)] += alpha * (loss_A_factor * treasure_TD + loss_B_factor * r_TD)

                    time_TDs.append(time_TD)
                    treasure_TDs.append(treasure_TD)
                    r_TDs.append(r_TD)
                    actions.append(ACTIONS[action])

                    if done:
                        # print("DONE")
                        break
                    state = next_state

                if i % 200 == 0:
                    self.target_time_Q = copy.deepcopy(self.time_Q)
                    self.target_treasure_Q = copy.deepcopy(self.treasure_Q)
                if i % num_train_rounds == 0:
                    print(f"Episodes:{j}\n"
                          f"Train Round:{i}--> Epsilon:{epsillon}\n"
                          f"Preference:{weights}\n"
                          f"Time TD:{np.mean(np.array(time_TDs))}\n"
                          f"Treasure TD:{np.mean(np.array(treasure_TDs))}\n"
                          f"R TD:{np.mean(np.array(r_TDs))}\n"
                          f"Action traj:{actions}\n"
                          f"Rewards:{rewards}")
                    print("==================================================")

    def generate_trajectory(self, policy_type=True, num_trajs=1000, visible=False, temperature=1.4,
                            indicated_weights=None):
        a_traj_dict = {}
        s_traj_dict = {}
        s_a_traj_dict = {}
        r_vec_traj_dict = {}
        r_traj_dict = {}
        if indicated_weights is None:
            indicated_weights = self.preference_list
        for weights in indicated_weights:
            a_traj_dict[weights] = []
            s_traj_dict[weights] = []
            s_a_traj_dict[weights] = []
            r_vec_traj_dict[weights] = []
            r_traj_dict[weights] = []

            sum_episode_rewards = np.zeros(2)
            for i in range(num_trajs):
                state = self.env.reset()
                done = False
                s_traj = []
                a_traj = []
                s_a_traj = []

                episode_rewards = np.zeros(2)
                episode_reward = 0
                while not done:
                    if policy_type == "deterministic":
                        action = self.envelope_epsilon_greedy(state, 0, weights)
                    elif policy_type == "stochastic":
                        action = self.soft_max_q(state, weights, temperature)
                    elif policy_type == "mix":
                        r = random.random()
                        # print(r)
                        if r > 0.1:
                            action = self.envelope_epsilon_greedy(state, 0, weights)
                        else:
                            action = self.soft_max_q(state, weights, temperature)

                    s = state[0] * 10 + state[1]
                    a_traj.append(action)
                    s_traj.append(s)
                    s_a_traj.append([s, action])

                    state, rewards, done = self.env.step(action)
                    reward = self.scalarise(rewards, weights)
                    episode_rewards += rewards
                    episode_reward += reward
                # if i % 100 == 0 and i > 100:
                #     print(f"traj:{num_trajs} generated")

                sum_episode_rewards += episode_rewards

                a_traj_dict[weights].append(np.array(a_traj))
                s_traj_dict[weights].append(np.array(s_traj))
                s_a_traj_dict[weights].append(np.array(s_a_traj))
                r_vec_traj_dict[weights].append(np.array(episode_rewards))
                r_traj_dict[weights].append(np.array([episode_reward]))

            if visible:
                print(f"mean reward:{sum_episode_rewards / num_trajs}")
                print("============================================================================")

        return a_traj_dict, s_traj_dict, s_a_traj_dict, r_vec_traj_dict, r_traj_dict


if __name__ == '__main__':
    env = DeepSeaTreasureEnvironment()
    q_agent = MOTabularAgent(env)
    state = env.reset()
    print(q_agent.envelope_epsilon_greedy(state=state, epsillon=0, weight=np.array([0.1, 0.9])))
    # profile = line_profiler.LineProfiler(q_agent.soft_max_q)
    # profile.enable()
    # q_agent.soft_max_q(state=state, weight=(0.9, 0.1))
    # profile.disable()
    # profile.print_stats()
