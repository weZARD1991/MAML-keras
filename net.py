# -*- coding: utf-8 -*-
# @File : net.py
# @Author: Runist
# @Time : 2020/7/6 16:52
# @Software: PyCharm
# @Brief: 实现模型分类的网络，MAML与网络结构无关，重点在训练过程

from tensorflow.keras import layers, activations, Model, optimizers, models
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import config as cfg


class MAMLmodel(Model):
    def __init__(self, num_classes, width=cfg.width, height=cfg.height, channel=cfg.channel):
        super(MAMLmodel, self).__init__()
        self.Conv2D_1 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu",
                                         input_shape=[width, height, channel])
        self.Conv2D_2 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")
        self.MaxPooling1 = layers.MaxPool2D(pool_size=2)

        self.Conv2D_3 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")
        self.Conv2D_4 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")
        self.MaxPooling2 = layers.MaxPool2D(pool_size=2)

        self.Conv2D_5 = layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")
        self.Conv2D_6 = layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")
        self.MaxPooling3 = layers.MaxPool2D(pool_size=2)

        self.Flatten = layers.Flatten()
        self.Dense_1 = layers.Dense(128, activation='relu')
        self.Dense_2 = layers.Dense(num_classes)

    def forward(self, inputs):
        x = self.Conv2D_1(inputs)
        x = self.Conv2D_2(x)
        x = self.MaxPooling1(x)

        x = self.Conv2D_3(x)
        x = self.Conv2D_4(x)
        x = self.MaxPooling2(x)

        x = self.Conv2D_5(x)
        x = self.Conv2D_6(x)
        x = self.MaxPooling3(x)

        x = self.Flatten(x)
        x = self.Dense_1(x)
        x = self.Dense_2(x)

        return x


def MAML_model(num_classes, width=cfg.width, height=cfg.height, channel=cfg.channel):
    model = models.Sequential([
        layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu",
                                         input_shape=[width, height, channel]),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2, strides=2),

        layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2, strides=2),

        layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2, strides=2),

        layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2, strides=2),

        layers.Flatten(),
        layers.Dense(num_classes),
    ])
    return model
