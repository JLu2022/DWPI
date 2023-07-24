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


class ImageDST(object):
    def __init__(self, visualization=False, submarine_pos=None):
        # CCDST
        row_1 = (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0)
        row_2 = (1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0)
        row_3 = (0, 34, -1, -1, -1, -1, -1, -1, -1, -1, 0)
        row_4 = (0, 0, 58, -1, -1, -1, -1, -1, -1, -1, 0)
        row_5 = (0, 0, 0, 78, 86, 92, -1, -1, -1, -1, 0)
        row_6 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0)
        row_7 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0)
        row_8 = (0, 0, 0, 0, 0, 0, 112, 116, -1, -1, 0)
        row_9 = (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0)
        row_10 = (0, 0, 0, 0, 0, 0, 0, 0, 122, -1, 0)
        row_11 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 124, 0)

        # row_1 = (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0)
        # row_2 = (0.7, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0)
        # row_3 = (0, 8.2, -1, -1, -1, -1, -1, -1, -1, -1, 0)
        # row_4 = (0, 0, 11.5, -1, -1, -1, -1, -1, -1, -1, 0)
        # row_5 = (0, 0, 0, 14.0, 15.1, 16.1, -1, -1, -1, -1, 0)
        # row_6 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0)
        # row_7 = (0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0)
        # row_8 = (0, 0, 0, 0, 0, 0, 19.6, 20.3, -1, -1, 0)
        # row_9 = (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0)
        # row_10 = (0, 0, 0, 0, 0, 0, 0, 0, 22.4, -1, 0)
        # row_11 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 23.7, 0)
        self.background_map = (row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8, row_9, row_10, row_11)
        self.img_map = [list(self.background_map[0]), list(self.background_map[1]), list(self.background_map[2]),
                        list(self.background_map[3]), list(self.background_map[4]), list(self.background_map[5]),
                        list(self.background_map[6]), list(self.background_map[7]), list(self.background_map[8]),
                        list(self.background_map[9]), list(self.background_map[10])]

        # self.image = image

        if submarine_pos is None:
            self.submarine_pos = [0, 0]
        else:
            self.submarine_pos = submarine_pos
        self.row = self.submarine_pos[0]
        self.col = self.submarine_pos[1]
        # print(f"submarine:{submarine_pos}\trow:{self.row}\tcol:{self.col}")
        self.num_row = len(self.background_map)
        self.num_col = len(self.background_map[0])
        self.observation_space = (self.num_row, self.num_col, 3)
        self.action_space = 4
        self.add_submarine()
        self.visualization = visualization
        self.preference_token = "No pref"
        self.image_path = "DST/" + str(self.preference_token) + "/"
        self.energy = 100

    def render_map(self, img_map):
        image = np.zeros(self.observation_space)
        for row in range(0, self.num_row):
            for col in range(0, self.num_col):
                if img_map[row][col] == 0:
                    image[row][col][0] = 0
                    image[row][col][1] = 0
                    image[row][col][2] = 0
                elif img_map[row][col] == -1:  # available
                    image[row][col][0] = 0
                    image[row][col][1] = 191
                    image[row][col][2] = 255
                elif img_map[row][col] == 99:
                    image[row][col][0] = 255
                    image[row][col][1] = 0
                    image[row][col][2] = 0
                else:
                    image[row][col][0] = 255
                    image[row][col][1] = 255
                    image[row][col][2] = 0
        return image

    def show_available_position(self, row_index, num_sample):
        available_position = []
        # for row in range(self.num_col):
        count = 0
        while count < num_sample:
            col = random.choice(range(self.num_col))
            if self.img_map[row_index][col] == -1 or self.img_map[row_index][col] == 99:
                available_position.append([row_index, col])
                count += 1
        return available_position

    def reset(self, put_submarine=True):

        self.row = self.submarine_pos[0]
        self.col = self.submarine_pos[1]
        self.img_map = [list(self.background_map[0]), list(self.background_map[1]), list(self.background_map[2]),
                        list(self.background_map[3]), list(self.background_map[4]), list(self.background_map[5]),
                        list(self.background_map[6]), list(self.background_map[7]), list(self.background_map[8]),
                        list(self.background_map[9]), list(self.background_map[10])]
        if put_submarine:
            self.add_submarine()
        image = self.render_map(self.img_map)
        image /= 255
        self.energy = 100
        return image

    def add_submarine(self):
        self.img_map[self.row][self.col] = 99

    def step(self, action, episode=1001):  # 0:up 1:down 2:left 3:right
        self.img_map = [list(self.background_map[0]), list(self.background_map[1]), list(self.background_map[2]),
                        list(self.background_map[3]), list(self.background_map[4]), list(self.background_map[5]),
                        list(self.background_map[6]), list(self.background_map[7]), list(self.background_map[8]),
                        list(self.background_map[9]), list(self.background_map[10])]
        rewards = np.zeros(2)
        rewards[0] = -1
        terminal = False

        self.energy -= 1
        if action == 0 and self.row > 0 and not self.background_map[self.row - 1][self.col] == 0:
            self.row = self.row - 1
        if action == 1 and self.row < 10 and not self.background_map[self.row + 1][self.col] == 0:
            self.row = self.row + 1  # 0 Down, 1 Right
        elif action == 2 and self.col > 0 and not self.background_map[self.row][self.col - 1] == 0:
            self.col = self.col - 1
        elif action == 3 and self.col < 9 and not self.background_map[self.row][self.col + 1] == 0:
            self.col = self.col + 1

        if not self.background_map[self.row][self.col] == 0 and not self.background_map[self.row][self.col] == -1:
            rewards[1] = self.background_map[self.row][self.col]
            terminal = True
        if self.energy <= 0:
            terminal = True

        self.add_submarine()
        image = self.render_map(self.img_map)
        image /= 255
        position = self.row*11+self.col
        return rewards, image, terminal, position

    def visualize(self):
        my_ticks_x = np.arange(0, self.num_col, 1)
        my_ticks_y = np.arange(0, self.num_row, 1)
        plt.xticks(my_ticks_x)
        plt.yticks(my_ticks_y)
        plt.imshow(self.img_map)
        plt.savefig("try" + ".png")

    def set_visualization(self, visualization=False):
        self.visualization = visualization

    def set_preference(self, preference_token):
        self.preference_token = preference_token
        self.image_path = "/content/drive/MyDrive/Colab Notebooks/Source Code/AAMAS-2023/Route/" + str(
            self.preference_token) + "/"

    def check_forbidden(self, action, player_pos):
        row = player_pos[0]
        col = player_pos[1]
        if action == "up" and row > 0 and not self.background_map[row - 1][col] == 0:
            return False
        if action == "down" and row < 10 and not self.background_map[row + 1][col] == 0:
            return False
        elif action == "left" and col > 0 and not self.background_map[row][col - 1] == 0:
            return False
        elif action == "right" and col < 9 and not self.background_map[row][col + 1] == 0:
            return False
        else:
            return True

    def calculate_pos(self, player_pos):
        row = player_pos[0]
        col = player_pos[1]
        return (row * self.num_col + col)/100

    def calculate_reward(self, player_pos):
        row = player_pos[0]
        col = player_pos[1]
        rewards = np.zeros(2)
        if not self.background_map[row][col] == 0 and not self.background_map[row][col] == -1:
            rewards += np.array([-1, self.background_map[row][col]])
        else:
            rewards += np.array([-1, 0])
        return rewards/100

    def get_settings(self,action=None):
        return [self.img_map, {-1: (0, 191, 255),
                               0: (0, 0, 0),
                               99: (0, 191, 255),
                               1: (255, 255, 0),
                               34: (255, 255, 0),
                               58: (255, 255, 0),
                               78: (255, 255, 0),
                               86: (255, 255, 0),
                               92: (255, 255, 0),
                               112: (255, 255, 0),
                               116: (255, 255, 0),
                               122: (255, 255, 0),
                               124: (255, 255, 0)}]

    def check_terminal(self, player_pos):
        row = player_pos[0]
        col = player_pos[1]
        if not self.background_map[row][col] == 0 and not self.background_map[row][col] == -1:
            return True
        else:
            return False


if __name__ == '__main__':
    dst_env = ImageDST(visualization=True)
    dst_env.reset(put_submarine=False)
    print(dst_env.show_available_position())
    print(dst_env.img_map)
    positions = []
    train_set = []
    for row in range(11):
        for col in range(11):
            if dst_env.img_map[row][col] == -1:
                positions.append([row, col])
    for position in positions:
        dst_env.reset(put_submarine=False)
        dst_env.row = position[0]
        dst_env.col = position[1]
        dst_env.add_submarine()
        train_set.append(np.array([dst_env.render_map(dst_env.img_map).flatten() / 255]))
        img = dst_env.render_map(dst_env.img_map)
        plt.imshow(img)
        plt.show(block=False)
        plt.pause(0.4)
        plt.close()

    train_set = np.array(train_set)
    np.save("C://PhD//2023//Inverse_MORL//MOA2C//AE_dataset//Eye_train.npy", train_set)
    #
    # plt.imshow(img_map)
    # plt.show()
    # actions = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #            1, 1, 1, 1]
    # trainset = []
    # for i in range(100):
    #     action = np.random.choice([0, 1, 2, 3])
    #     rewards, img_map, terminal = dst_env.step(action)
    #     trainset.append(np.array([img_map.flatten()]))
    #     # plt.imshow(img_map)
    #     # plt.show(block=False)
    #     # plt.pause(0.4)
    #     # plt.close()
    #     print("rewards:{},terminal:{}".format(rewards, terminal))
    #     print("-------------------------------------------------")
    #     if terminal:
    #         dst_env.reset()
    # trainset = np.array(trainset)
    # np.save("C://PhD//2023//Inverse_MORL//MOA2C//AE_dataset//Eye_train.npy", trainset)
    #
    # dst_env.reset()
    # test_set_actions = [3, 3, 3, 1, 1, 3, 1, 3]
    # test_set = []
    # for i in range(len(test_set_actions)):
    #
    #     rewards, img_map, terminal = dst_env.step(test_set_actions[i])
    #     test_set.append(np.array([img_map.flatten()]))
    #     # plt.imshow(img_map)
    #     # plt.show(block=False)
    #     # plt.pause(0.4)
    #     # plt.close()
    #     print("rewards:{},terminal:{}".format(rewards, terminal))
    #     print("-------------------------------------------------")
    #     if terminal:
    #         break
    # test_set = np.array(test_set)
    # np.save("C://PhD//2023//Inverse_MORL//MOA2C//AE_dataset//Eye_test.npy", test_set)
    # print(test_set)
