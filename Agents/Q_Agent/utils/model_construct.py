from keras import layers
from keras.layers import Input, Concatenate, Dropout, Dense, Flatten, Conv2D, Reshape
from keras.models import Model
from keras.losses import mse
import keras
import tensorflow as tf
import numpy as np


class QNN(Model):
    def __init__(self, hidden_dim, out_dim, name="Q_"):
        super().__init__()
        self.dense1 = layers.Dense(16, activation="relu")
        # self.drop_out_0 = layers.Dropout(0.2, name="drop_0")
        self.dense2 = layers.Dense(32, activation="relu")
        # self.drop_out_1 = layers.Dropout(0.2, name="drop_1")
        self.dense3 = layers.Dense(16, activation="relu")
        # self.drop_out_2 = layers.Dropout(0.2, name="drop_2")
        self.dense4 = layers.Dense(8, activation="relu")
        self.dense5 = layers.Dense(out_dim)
        self.model_name = name

    def call(self, x, trainable=True, mask=None):
        x = self.dense1(x)
        # x = self.drop_out_0(x)
        x = self.dense2(x)
        # x = self.drop_out_1(x)
        x = self.dense3(x)
        # x = self.drop_out_2(x)
        x = self.dense4(x)
        out = self.dense5(x)
        return out
