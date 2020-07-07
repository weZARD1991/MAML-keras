# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2020/7/7 10:06
# @Software: PyCharm
# @Brief: 数据读取脚本

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend


# Other dependencies
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproduction
np.random.seed(333)


def read_csv(csv_path):
    """
    读取包含图片名和标签的csv
    :param csv_path:
    :return:
    """
    csv = pd.read_csv(csv_path)

    image_list = list("./data/images/" + csv.iloc[:, 0])

    label_list = pd.factorize(csv.iloc[:, 1])
    label_list = label_list[0].tolist()

    return image_list, label_list


def process(img_path, label, class_num=64):

    """
    对数据集批量处理的函数，传给map，给图片加一个左右翻转的操作，
    然后还要按照标准归一化操作-0.5再缩放
    :param img_path: 必须有的参数，图片路径
    :param label: 必须有的参数，图片标签（都是和dataset的格式对应）
    :param class_num: 类别数量
    :return: 单个图片和分类
    """
    label = tf.one_hot(label, depth=class_num)

    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    # 将unit8转为float32
    image = tf.dtypes.cast(image, tf.float32)
    image = tf.image.resize(image, [224, 224])

    # image = tf.image.random_flip_left_right(image)

    return image, label


def make_datasets(image, label, batch_size, mode="train"):
    dataset = tf.data.Dataset.from_tensor_slices((image, label))
    if mode == "train":
        dataset = dataset.shuffle(buffer_size=len(label))

    dataset = dataset.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


class SinusoidGenerator:
    """
    正弦信号发生器。
    p(T)是连续的，其中振幅变化范围在[0.1,5.0]内，相位变化范围在[0,kw]内。
    """
    def __init__(self, batch_size=10, amplitude=None, phase=None):
        """

        :param batch_size:
        :param amplitude: 振幅，如果为None，就随机从[0.1, 5.0]之间均匀采样
        :param phase:
        """
        self.batch_size = batch_size
        self.amplitude = amplitude if amplitude else np.random.uniform(0.1, 5.0)
        self.phase = phase if phase else np.random.uniform(0, np.pi)
        self.sampled_points = None
        self.x = self._sample_x()

    def _sample_x(self):
        return np.random.uniform(-5, 5, self.batch_size)

    def f(self, x):
        """
        正弦波
        :param x: w
        :return: sinewave
        """
        return self.amplitude * np.sin(x - self.phase)

    def batch(self, x=None, force_new=False):
        """
        返回一个大小为K的批处理。它还更改了x的shape，为其添加了批处理尺寸
        :param x:如果给定y，则基于此数据生成批处理数据。默认为无。如果没有，则使用`self.x`
        :param force_new: 如果x为None,可以选择是否采用随机均匀采样
        :return:
        """
        if x is None:
            if force_new:
                x = self._sample_x()
            else:
                x = self.x
        y = self.f(x)
        return x[:, None], y[:, None]

    def equally_spaced_samples(self, batch_size=None):
        """
        返回batch_size个等距样本。
        :param batch_size: 生成sinewave 序列的间距
        :return:
        """
        if batch_size is None:
            batch_size = self.batch_size
        return self.batch(x=np.linspace(-5, 5, batch_size))


def plot(data, *args, **kwargs):
    x, y = data

    return plt.plot(x, y, *args, **kwargs)


def generate_dataset(batch_size, train_size=20000, test_size=10):
    """
    生成训练并测试数据集。数据集由能够一次提供一批（`K`）元素的SinusoidGenerator组成。
    :param batch_size:
    :param train_size: 训练集的个数
    :param test_size: 测试集的个数
    :return:
    """

    def _generate_dataset(size):
        return [SinusoidGenerator(batch_size=batch_size) for _ in range(size)]

    return _generate_dataset(train_size), _generate_dataset(test_size)




