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

