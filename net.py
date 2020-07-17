# -*- coding: utf-8 -*-
# @File : net.py
# @Author: Runist
# @Time : 2020/7/6 16:52
# @Software: PyCharm
# @Brief: 实现模型分类的网络，MAML与网络结构无关，重点在训练过程

from tensorflow.keras import layers, activations, losses, Model, optimizers, models
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import config as cfg


class MAMLmodel(Model):
    def __init__(self, num_classes):
        super(MAMLmodel, self).__init__()
        self.Conv2D_1 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu",
                                         input_shape=[224, 224, 3])
        self.Conv2D_2 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")

        self.Conv2D_3 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")
        self.Conv2D_4 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")

        self.Flatten = layers.Flatten()
        self.Dense_1 = layers.Dense(128, activation='relu')
        self.Dense_2 = layers.Dense(num_classes)

    def forward(self, inputs):
        x = self.Conv2D_1(inputs)
        x = self.Conv2D_2(x)

        x = self.Conv2D_3(x)
        x = self.Conv2D_4(x)

        x = self.Flatten(x)
        x = self.Dense_1(x)
        x = self.Dense_2(x)

        return x


def np_to_tensor(numpy_objs):
    """
    将numpy转成tensor
    :param numpy_objs: numpy列表
    :return:
    """
    return (tf.convert_to_tensor(obj, dtype=tf.float32) for obj in numpy_objs)


def compute_loss(y_true, y_pred):
    """
    计算loss
    :param y_true: 模型
    :param y_pred:
    :return:
    """
    mse = losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    # mse = K.mean(losses.mean_squared_error(y_true, y_pred))

    return mse


def copy_model(model, x):
    """
    将权值拷贝出来到新的模型上
    :param model: 要被拷贝的模型
    :param x: 输入的task,这用于运行前向传递，以将图形的权重添加为变量。
    :return: 接受了新权值的模型
    """
    copied_model = MAMLmodel(cfg.num_classes)

    # 如果我们不执行此步骤，则权重不会“初始化”，并且不会计算梯度。
    copied_model.forward(x)
    copied_model.set_weights(model.get_weights())

    return copied_model


