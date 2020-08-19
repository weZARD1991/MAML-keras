# -*- coding: utf-8 -*-
# @File : net.py
# @Author: Runist
# @Time : 2020/7/6 16:52
# @Software: PyCharm
# @Brief: 实现模型分类的网络，MAML与网络结构无关，重点在训练过程

from tensorflow.keras import layers, activations, Model, optimizers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import config as cfg


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


class MAMLmodel(Model):

    def __init__(self, num_classes, width=cfg.width, height=cfg.height, channel=cfg.channel):
        super().__init__()

        self.conv2d_1 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu",
                                      input_shape=[width, height, channel])
        self.bn1 = layers.BatchNormalization()
        self.max_pool1 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv2d_2 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.max_pool2 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv2d_3 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")
        self.bn3 = layers.BatchNormalization()
        self.max_pool3 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv2d_4 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")
        self.bn4 = layers.BatchNormalization()
        self.max_pool4 = layers.MaxPool2D(pool_size=2, strides=2)

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_classes)

    def call(self, inputs):

        x = self.conv2d_1(inputs)
        x = self.bn1(x)
        x = self.max_pool1(x)

        x = self.conv2d_2(x)
        x = self.bn2(x)
        x = self.max_pool2(x)

        x = self.conv2d_3(x)
        x = self.bn3(x)
        x = self.max_pool3(x)

        x = self.conv2d_4(x)
        x = self.bn4(x)
        x = self.max_pool4(x)

        x = self.flatten(x)
        x = self.dense(x)

        return x


def model_copy(target_model):
    copy_model = MAMLmodel(cfg.n_way)
    copy_model.build(input_shape=(None, cfg.height, cfg.width, cfg.channel))

    copy_model.set_weights(target_model.get_weights())

    return copy_model

