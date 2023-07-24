import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import Dense
from collections import deque  # Used for replay buffer and reward tracking
from datetime import datetime  # Used for timing script
import time


class ItemGathering(object):
    def __init__(self, visualization=False):
        """
        Green: 0; Red: 1; Yellow: 2; White: 3; Blue: 4; Black: 5;
        :param visualization:
        """
        row_1 = (5, 5, 5, 5, 5, 5, 5, 5)
        row_2 = (5, 5, 5, 5, 5, 5, 5, 5)
        row_3 = (5, 5, 5, 5, 5, 5, 5, 5)
        row_4 = (5, 5, 5, 5, 5, 5, 5, 5)
        row_5 = (5, 5, 5, 5, 5, 5, 5, 5)
        row_6 = (5, 5, 5, 5, 5, 5, 5, 5)
        row_7 = (5, 5, 5, 5, 5, 5, 5, 5)
        row_8 = (5, 5, 5, 5, 5, 5, 5, 5)

        self.background_map = (row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8)
        # fixed position
        self.green_block_position = [[4, 3], [4, 5], [5, 2]]
        self.red_block_position = [[2, 3], [2, 5], [4, 2]]
        self.yellow_block_position = [[2, 2], [5, 4]]
        # random_position
        # add multiple agents to the environment
        # self.agents = agents
        #
        # self.agents_positions = [[7,0],[0,7]]

        point_set = self.sample_random_pos(8)
        self.green_block_position = [list(point_set.pop()), list(point_set.pop()), list(point_set.pop())]
        self.red_block_position = [list(point_set.pop()), list(point_set.pop()), list(point_set.pop())]
        self.yellow_block_position = [list(point_set.pop()), list(point_set.pop())]
        self.treasure_status = {"green": [True, True, True], "red": [True, True, True], "yellow": [True, True]}

        self.row = 7
        self.col = 0

        self.energy = 50
        self.img_map = [list(self.background_map[0]), list(self.background_map[1]), list(self.background_map[2]),
                        list(self.background_map[3]), list(self.background_map[4]), list(self.background_map[5]),
                        list(self.background_map[6]), list(self.background_map[7])]
        self.observation_space = (8, 8, 3)
        self.fixed_agent_pos = [0, 7]
        self.action_space = 4  # include stay
        self.add_treasure()
        self.add_agent()

    def sample_random_pos(self, number_points=8):
        point_set = set()
        while len(point_set) < number_points:
            point = np.random.choice(8, 2)
            point = tuple(point)
            if not point == (7, 0) and not point == (0, 7):
                point_set.add(point)
        return point_set

    def reset(self):
        # green, red, yellow
        self.treasure_status = {"green": [True, True, True], "red": [True, True, True], "yellow": [True, True]}
        self.energy = 50
        self.row = 7
        self.col = 0
        self.fixed_agent_pos = [0, 7]
        # self.agents_positions = [[7, 0], [0, 7]]
        point_set = self.sample_random_pos(8)
        self.green_block_position = [list(point_set.pop()), list(point_set.pop()), list(point_set.pop())]
        self.red_block_position = [list(point_set.pop()), list(point_set.pop()), list(point_set.pop())]
        self.yellow_block_position = [list(point_set.pop()), list(point_set.pop())]
        self.img_map = [list(self.background_map[0]), list(self.background_map[1]), list(self.background_map[2]),
                        list(self.background_map[3]), list(self.background_map[4]), list(self.background_map[5]),
                        list(self.background_map[6]), list(self.background_map[7])]
        self.add_treasure()
        self.add_agent()
        self.add_fixed_agent()
        self.img_map = self.render_map(self.img_map)
        self.img_map /= 255
        return self.img_map

    def render_map(self, img_map):
        image = np.zeros((8, 8, 3))  # Green: 0; Red: 1; Yellow: 2; White: 3; Blue: 4; Black: 5;
        for row in range(0, 8):
            for col in range(0, 8):
                if img_map[row][col] == 0:  # Green: 0
                    image[row][col][0] = 0
                    image[row][col][1] = 128
                    image[row][col][2] = 0
                elif img_map[row][col] == 1:  # Red: 1
                    image[row][col][0] = 255
                    image[row][col][1] = 0
                    image[row][col][2] = 0
                elif img_map[row][col] == 2:  # Yellow: 2
                    image[row][col][0] = 255
                    image[row][col][1] = 255
                    image[row][col][2] = 0
                elif img_map[row][col] == 3:  # White: 3
                    image[row][col][0] = 255
                    image[row][col][1] = 255
                    image[row][col][2] = 255
                elif img_map[row][col] == 4:  # Blue: 4
                    image[row][col][0] = 0
                    image[row][col][1] = 0
                    image[row][col][2] = 255
                elif img_map[row][col] == 5:  # Black: 5
                    image[row][col][0] = 0
                    image[row][col][1] = 0
                    image[row][col][2] = 0
                elif img_map[row][col] == 6:  # Fixed Agent Pink: 6
                    image[row][col][0] = 255
                    image[row][col][1] = 105
                    image[row][col][2] = 180
        return image

    def get_euclidean_dist(self, pos_1, pos_2):
        return abs(pos_1[0] - pos_2[0]) + abs(pos_1[1] - pos_2[1])

    def fixed_agent_move(self, step):
        # print("fix move")
        index = 0
        min_dist = np.inf

        for i in range(len(self.red_block_position)):
            if self.treasure_status["red"][i]:
                dist = self.get_euclidean_dist(self.fixed_agent_pos, self.red_block_position[i])
                # print(dist)
                if dist < min_dist:
                    min_dist = dist
                    index = i
        # print("closest is:{},mini_dist:{}".format(index, min_dist))
        need_down = self.fixed_agent_pos[0] - self.red_block_position[index][0]
        need_left = self.fixed_agent_pos[1] - self.red_block_position[index][1]
        # print(f"need_down:{need_down}\tneed_left:{need_left}")
        rand = random.random()
        if rand < 0.5:
            if abs(need_down) > 0:
                self.fixed_agent_pos = [self.fixed_agent_pos[0] + int(-need_down / abs(need_down)),
                                        self.fixed_agent_pos[1]]
            elif abs(need_left) > 0:
                self.fixed_agent_pos = [self.fixed_agent_pos[0],
                                        self.fixed_agent_pos[1] - int(need_left / abs(need_left))]
        else:
            if abs(need_left) > 0:
                self.fixed_agent_pos = [self.fixed_agent_pos[0],
                                        self.fixed_agent_pos[1] - int(need_left / abs(need_left))]
            elif abs(need_down) > 0:
                self.fixed_agent_pos = [self.fixed_agent_pos[0] + int(-need_down / abs(need_down)),
                                        self.fixed_agent_pos[1]]
        # print(" ---------------------- ")

    def add_fixed_agent(self):
        # print(self.fixed_agent_pos[0], self.fixed_agent_pos[1])
        self.img_map[self.fixed_agent_pos[0]][self.fixed_agent_pos[1]] = 6

    def add_treasure(self):
        for i in range(len(self.treasure_status["green"])):
            if self.treasure_status["green"][i]:
                self.img_map[self.green_block_position[i][0]][self.green_block_position[i][1]] = 0
        for j in range(len(self.treasure_status["red"])):
            if self.treasure_status["red"][j]:
                self.img_map[self.red_block_position[j][0]][self.red_block_position[j][1]] = 1

        for k in range(len(self.treasure_status["yellow"])):
            if self.treasure_status["yellow"][k]:
                self.img_map[self.yellow_block_position[k][0]][self.yellow_block_position[k][1]] = 2

    def add_agent(self):
        self.img_map[self.row][self.col] = 4

    def step(self, action):  # 0:up 1:down 2:left 3:right
        if action is not None:
            # print(f"step,actions:{action}")
            self.img_map = [list(self.background_map[0]), list(self.background_map[1]), list(self.background_map[2]),
                            list(self.background_map[3]), list(self.background_map[4]), list(self.background_map[5]),
                            list(self.background_map[6]), list(self.background_map[7])]
            terminal = False
            rewards = np.zeros(6)
            rewards[0] -= 1  # time penalty

            self.energy -= 1
            original_position = [self.row, self.col]
            if action == 0:
                self.row = self.row - 1
            elif action == 1:
                self.row = self.row + 1
            elif action == 2:
                self.col = self.col - 1
            elif action == 3:
                self.col = self.col + 1

            self.fixed_agent_move(step=49 - self.energy)

            if self.fixed_agent_pos == [self.row, self.col]:
                [self.row, self.col] = original_position

            rewards[5] += self.check_collection(self.fixed_agent_pos, self.red_block_position, "red")
            self.check_collection(self.fixed_agent_pos, self.green_block_position, "green")
            self.check_collection(self.fixed_agent_pos, self.yellow_block_position, "yellow")

            rewards[2] += self.check_collection(pos=[self.row, self.col], target=self.green_block_position,
                                                target_name="green")
            rewards[3] += self.check_collection(pos=[self.row, self.col], target=self.red_block_position,
                                                target_name="red")
            rewards[4] += self.check_collection(pos=[self.row, self.col], target=self.yellow_block_position,
                                                target_name="yellow")

            if self.check_wall(self.row, self.col):
                rewards[1] -= 5
                [self.row, self.col] = original_position

            self.add_fixed_agent()
            self.add_treasure()
            self.add_agent()

            if (self.treasure_status["green"] == [False, False, False] and self.treasure_status["red"] == [False, False,
                                                                                                           False]
                and self.treasure_status["yellow"] == [False, False]) or self.energy == 0:
                terminal = True

            self.img_map = self.render_map(self.img_map)
            self.img_map /= 255
            # print(self.fixed_agent_pos)
            position = self.row * 8 + self.col
            return rewards, self.img_map, terminal, position
        else:
            return None, self.img_map, None, None

    def check_wall(self, row, col):
        if row < 0 or row > 7 or col < 0 or col > 7:
            return True
        else:
            return False

    def check_collection(self, pos, target, target_name):
        # print(len(target))
        collected = False
        for i in range(0, len(target)):
            if pos == target[i] and self.treasure_status[target_name][i]:
                self.treasure_status[target_name][i] = False
                collected = True
        return collected

    def get_settings(self, action=None):
        self.step(action)
        return [self.img_map, {0: (0, 128, 0),
                               1: (255, 0, 0),
                               2: (255, 255, 0),
                               3: (255, 255, 255),
                               4: (0, 0, 255),
                               5: (0, 0, 0),
                               6: (255, 105, 180)}]


if __name__ == '__main__':
    mo_traffic = ItemGathering(visualization=True)
    img_map = mo_traffic.reset()
    print(img_map)
    plt.imshow(img_map)
    plt.show()
    for i in range(18):
        actions = [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        rewards, img_map, terminal = mo_traffic.step(actions[i])
        plt.imshow(img_map)
        plt.show(block=False)
        plt.pause(0.4)
        plt.close()
        print("rewards:{},terminal:{}".format(rewards, terminal))
        print("-------------------------------------------------")
        if terminal:
            break
