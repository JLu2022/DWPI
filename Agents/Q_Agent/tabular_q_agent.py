import csv
import random

import line_profiler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Environments.DeepSeaTreasure.DiscreteDST import DeepSeaTreasureEnvironment
from scipy.special import softmax

ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right"}
Training_visualization_path = "C://Users//19233436//PycharmProjects//DWPI//Train_Agent//Train Result//discreteDST_Result//Learning_curve//"


class Tabular_Q_Agent:
    def __init__(self, env):
        self.env = env
        self.actions = [i for i in range(len(env.actions))]
        self.preference_list = [(np.round(w0 / 100, 2), np.round(1 - w0 / 100, 2)) for w0 in range(0, 101)]
        # self.preference_list = [np.array([0.06, 0.94])]
        self.initialise_q_values()

    def vanilla_epsilon_greedy(self, state, epsilon, weight_str, temperature=1.4, scene=None):
        if np.random.rand() < epsilon:
            return np.random.choice(self.actions), None
        else:
            return np.argmax(self.Q_dict[weight_str][state]), None

    def soft_policy(self, state, epsilon, weight_str, temperature=1.4, scene=None, truncated_thres=2):
        q_values = self.Q_dict[weight_str][state]
        max_2_index = q_values.argsort()[-truncated_thres:]
        mask = np.zeros(4)
        q_values -= np.mean(q_values)
        for i in max_2_index:
            mask[i] = 1

        q_masked = mask * q_values
        q_masked_sqr = q_masked ** temperature
        for i in range(len(q_masked_sqr)):
            if q_masked_sqr[i] == 0:
                q_masked_sqr[i] = -np.inf
        q_probs = softmax(q_masked_sqr)
        action = np.random.choice([0, 1, 2, 3], p=q_probs)

        return action, None

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

    def q_learning(self, episodes):
        epsillon0 = 0.8
        alpha = 0.2
        gamma = 1

        self.stats_dict = {}
        self.initialise_q_values()
        for weights in self.preference_list:

            stats = []
            for i in range(episodes):
                state = self.env.reset()
                rs = 0

                epsillon = max(epsillon0 - i / episodes, 0.0001)
                # epsillon = 0.9999**i

                while True:
                    action, _ = self.vanilla_epsilon_greedy(state, epsillon, weights)
                    next_state, rewards, done, _ = self.env.step(action)
                    reward = self.scalarise(rewards, weights)
                    rs += reward
                    self.Q_dict[weights][(*state, action)] += alpha * (
                            reward + gamma * np.max(self.Q_dict[weights][next_state]) -
                            self.Q_dict[weights][(*state, action)])
                    if done:
                        break
                    state = next_state
                stats.append([i, rs])
            key = tuple(np.round(weights, 4))
            # if i%10 == 0:

            self.stats_dict[key] = [np.array(stats), self.Q_dict[weights].copy()]
            self.plot_learning_curve(self.stats_dict[key][0], key, weights)

    # , episode, epsilon, weight_i, mode="envelope", scene="train"

    def generate_experience(self, episode, epsilon, weight_i, scene="train",
                            deterministic_ratio=1, num_trajs=1, max_state_traj_len=100, temperature=1.4,
                            reward_mask=False, truncated_thres=2):
        # print("generating")
        vec_r_traj = []
        scalar_r_traj = []
        states_traj = []
        true_treasure_feedback = 0
        if reward_mask:
            state = self.env.reset()
            while True:
                action, token = self.vanilla_epsilon_greedy(state=state, epsilon=epsilon, weight_str=weight_i,
                                                            scene=scene)
                next_state, rewards, done,_ = self.env.step(action, episode=episode)

                state = next_state
                if done:
                    true_treasure_feedback = rewards[1]
                    # print(f"weight_i:{weight_i}\tgenerate_true feedback:{true_treasure_feedback}")
                    break

        i = 0
        while i < num_trajs:
            # print(f"inloop i:{i},num_traj:{num_trajs},reward_mask:{reward_mask}")

            state_traj = np.zeros(110)
            action_traj = []
            state = self.env.reset()
            episode_time_reward = 0
            episode_treasure_reward = 0
            episode_scalar_reward = 0
            done = False
            treasure_reward = 0
            while True:
                if np.random.rand() <= deterministic_ratio:
                    action, token = self.vanilla_epsilon_greedy(state=state, epsilon=epsilon, weight_str=weight_i,
                                                                scene=scene)
                else:
                    action, token = self.soft_policy(state=state, epsilon=epsilon, weight_str=weight_i,
                                                     temperature=temperature, scene=scene, truncated_thres=truncated_thres)

                next_state, rewards, done,_ = self.env.step(action, episode=episode)
                # print(state)

                time_reward = rewards[0]
                treasure_reward = rewards[1]
                scalar_reward = time_reward * weight_i[0] + treasure_reward * weight_i[1]

                episode_time_reward += time_reward
                episode_treasure_reward += treasure_reward
                episode_scalar_reward += scalar_reward

                state = next_state

                state_traj[state[0] * 10 + state[1]] += 1
                action_traj.append(action)
                if done:
                    break
            state_traj = np.array(state_traj)


            if reward_mask:
                if true_treasure_feedback == treasure_reward:
                    i += 1
                    vec_r_traj.append(np.array([episode_time_reward / 100, episode_treasure_reward / 100]))
                    scalar_r_traj.append(np.array([episode_scalar_reward / 100]))
                    states_traj.append(state_traj)

            else:
                i += 1
                vec_r_traj.append(np.array([episode_time_reward / 100, episode_treasure_reward / 100]))
                scalar_r_traj.append(np.array([episode_scalar_reward / 100]))
                states_traj.append(state_traj)
        return vec_r_traj, action_traj, states_traj


if __name__ == '__main__':
    env = DeepSeaTreasureEnvironment()
    q_agent = Tabular_Q_Agent(env)
    state = env.reset()
    profile = line_profiler.LineProfiler(q_agent.soft_policy)
    profile.enable()
    q_agent.soft_policy(state=state, weight=(0.9, 0.1))
    profile.disable()
    profile.print_stats()
