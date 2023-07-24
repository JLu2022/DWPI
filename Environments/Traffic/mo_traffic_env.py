import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np
import keyboard
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


class MOTraffic(object):
    def __init__(self, visualization=False):
        """
        Green: 0; Red: 1; Yellow: 2; White: 3; Blue: 4; Black: 5;
        :param visualization:
        """
        row_1 = (5, 2, 2, 2, 2, 2, 2, 5)
        row_2 = (5, 2, 2, 2, 2, 2, 2, 5)
        row_3 = (5, 3, 3, 2, 2, 2, 2, 5)
        row_4 = (5, 5, 2, 2, 2, 2, 2, 5)
        row_5 = (5, 5, 3, 3, 2, 2, 2, 5)
        row_6 = (5, 5, 5, 5, 2, 2, 2, 5)
        row_7 = (5, 5, 3, 3, 3, 3, 3, 5)
        row_8 = (5, 5, 5, 5, 5, 5, 5, 5)

        self.background_map = (row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8)
        self.car_positions = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [0, 6]]
        self.furthest_car_row = [1, 1, 3, 5, 5, 5]
        self.treasure_status = [1, 1]

        self.row = 7
        self.col = 0
        self.energy = 100
        self.img_map = [list(self.background_map[0]), list(self.background_map[1]), list(self.background_map[2]),
                        list(self.background_map[3]), list(self.background_map[4]), list(self.background_map[5]),
                        list(self.background_map[6]), list(self.background_map[7])]
        self.observation_space = (8, 8, 3)
        self.action_space = 4  # include stay
        self.add_cars()
        self.add_treasure()
        self.add_agent()

    def reset(self):
        self.car_positions = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [0, 6]]
        self.treasure_status = [1, 1]
        self.energy = 100
        self.row = 7
        self.col = 0
        self.img_map = [list(self.background_map[0]), list(self.background_map[1]), list(self.background_map[2]),
                        list(self.background_map[3]), list(self.background_map[4]), list(self.background_map[5]),
                        list(self.background_map[6]), list(self.background_map[7])]
        self.add_cars()
        self.add_treasure()
        self.add_agent()
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
        return image

    def car_move(self):
        car_moves = []
        for _ in range(len(self.car_positions)):
            car_moves.append(random.choice([-1, 1]))  # -1: UP; 1: DOWN
        for i in range(len(self.car_positions)):
            theoretical_position = car_moves[i] + self.car_positions[i][0]
            self.car_positions[i][0] = max(theoretical_position, 0)
            self.car_positions[i][0] = min(self.car_positions[i][0], self.furthest_car_row[i])

    def add_cars(self):
        for pos in self.car_positions:
            self.img_map[pos[0]][pos[1]] = 1

    def add_treasure(self):
        if self.treasure_status[0] == 1:
            self.img_map[0][0] = 0
        else:
            self.img_map[0][0] = 5
        if self.treasure_status[1] == 1:
            self.img_map[0][7] = 0
        else:
            # print("self.treasure_status[1],",self.treasure_status[1])
            self.img_map[0][7] = 5

    def add_agent(self):
        self.img_map[self.row][self.col] = 4

    def step(self, action):  # 0:up 1:down 2:left 3:right
        if action is not None:
            # print(f"step, action:{action}")
            self.img_map = [list(self.background_map[0]), list(self.background_map[1]), list(self.background_map[2]),
                            list(self.background_map[3]), list(self.background_map[4]), list(self.background_map[5]),
                            list(self.background_map[6]), list(self.background_map[7])]
            terminal = False
            rewards = np.zeros(5)
            rewards[0] -= 1
            original_position = [self.row, self.col]
            if action == 0:
                self.row = self.row - 1
            elif action == 1:
                self.row = self.row + 1
            elif action == 2:
                self.col = self.col - 1
            elif action == 3:
                self.col = self.col + 1

            if self.row == 0 and self.col == 0:
                rewards[1] += 1 * self.treasure_status[0]
                self.treasure_status[0] = 0
            if self.row == 0 and self.col == 7:
                rewards[1] += 1 * self.treasure_status[1]
                self.treasure_status[1] = 0

            if self.check_wall(self.row, self.col):
                rewards[4] -= 1
                [self.row, self.col] = original_position

            # if not walk on the road
            if not (self.col == 0 or self.col == 7 or self.row == 7 or (
                    self.col == 1 and (3 < self.row <= 7))):
                rewards[2] -= 1

            self.car_move()
            if [self.row, self.col] in self.car_positions:
                # print("collision_deteced:[", self.row, ", ", self.col, "]")
                rewards[3] -= 1
            self.add_cars()
            self.add_treasure()
            self.add_agent()
            self.energy -= 1
            # print("sefl.treasure_status",self.treasure_status==[0,0])
            if self.treasure_status == [0, 0] or self.energy == 0:
                terminal = True

            self.img_map = self.render_map(self.img_map)
            # print("position", [self.row, self.col])
            self.img_map /= 255
            position = self.row * 8 + self.col

            return rewards, self.img_map, terminal, position
        else:
            return None, None, None, None

    def check_wall(self, row, col):
        if row < 0 or row > 7 or col < 0 or col > 7 or (row == 2 and col in [1, 2]) or (row == 4 and col in [2, 3]) or (
                row == 6 and col in [2, 3, 4, 5, 6]):
            return True
        else:
            return False

    def get_settings(self,action=None):
        self.step(action)
        return [self.img_map, {0: (0, 128, 0),
                               1: (255, 0, 0),
                               2: (255, 255, 0),
                               3: (255, 255, 255),
                               4: (0, 0, 255),
                               5: (0, 0, 0)}]


if __name__ == '__main__':
    mo_traffic = MOTraffic(visualization=True)
    img_map = mo_traffic.reset()
    print(img_map)
