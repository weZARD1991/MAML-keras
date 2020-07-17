# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2020/7/7 10:06
# @Software: PyCharm
# @Brief: 数据读取脚本 -- 数据读取的流程：
# ---------------------------------------------
# 先由一系列函数生成图片路径，组成任务，此时是没有标签的
# 所以一个任务中的排序必须是[:n_way*k_shot]是“训练”图片
# [n_way*k_shot:]是“验证”图片，最后在训练的时候对一个任务生成对应的标签

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
import pandas as pd
import random
import numpy as np


def read_csv(csv_path, one_class_img=600):
    """
    读取包含图片名和标签的csv
    :param csv_path:
    :return:
    """
    csv = pd.read_csv(csv_path)
    image_label = []

    image_list = list("./data/images/" + csv.iloc[:, 0])

    num_class = len(image_list) // one_class_img    # 总共有几类
    classes = [[] for _ in range(num_class)]

    # label_list = pd.factorize(csv.iloc[:, 1])
    # label_list = label_list[0].tolist()

    # for img, label in zip(image_list, label_list):
    #     image_label.append([img, label])

    # 先按照类区分开
    for i in range(num_class):
        start = i * one_class_img
        end = (i+1) * one_class_img
        classes[i] = image_list[start: end]

    return classes


def get_meta_batch(dataset, meta_batch_size):
    """
    生成一个batch的任务，用于训练。将传入的列表中的数据组合成一个batch_size
    :param dataset:
    :param meta_batch_size: batch_size个任务组成一个meta_batch
    :return: 生成一个batch的任务
    """
    j = 0
    while len(dataset) > 0:
        batch_task = []
        j %= len(dataset)
        for i in range(meta_batch_size):
            try:
                data = process_one_task(dataset[j])
                data = tf.squeeze(data, axis=1)
                batch_task.append(data)

            except IndexError:
                return
            j += 1

        # 将他们组合到新的任务里
        yield tf.stack(batch_task)


def process_one_task(one_task):
    """
    对一个任务处理，对其中每一个图片进行读取
    :param one_task: 一个batch的任务[img_path, label]
    :return:
    """
    task = []

    for lines in one_task:
        img_path, label = lines.split()

        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        # 将unit8转为float32且归一化
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [224, 224])

        task.append([image])

    return task


def create_label(n_way, k_shot):
    """
    创建标签，生成一个0 - n_way的序列，每个元素重复k_shot次
    :param n_way:
    :param k_shot:
    :return:
    """
    return tf.convert_to_tensor(np.repeat(range(n_way), k_shot), dtype=tf.float32)


def task_split(classes: list, q_query=1, n_way=5, k_shot=1):
    """
    将各个分类下的img_path，按任务为单位分类。这个API是基于Mini-ImageNet下实现的，其中每个类只有600个
    为了均匀利用到所有数据，(q_query + k_shot) * n_way 要能被 图片数量整除
    。n-way * k-shot张图片用来给inner loop训练，n-way * query是给out loop去test
    dataset最终是 , shape = [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
    :param classes: shape为(class_num, img_num)的二位列表，存储了图片的路径
    :param q_query: query-set的数量
    :param n_way: 一个任务由几个类组成
    :param k_shot: support-set数量
    :return:
    """
    dataset = []
    # 这样计算循环次数的前提得是每个分类中图片数量相同，下面划分数据集的操作也是基于这个前提才能计算的
    # 总的循环数 = 图片总数 // 一个任务所包含图片的数量
    loop_num = len(classes) * len(classes[0]) // ((q_query + k_shot) * n_way)

    choose = [i for i in range(len(classes))]
    random.shuffle(choose)

    end = 0

    for _ in range(loop_num):
        # 用来存储一个任务的图片
        one_task = []
        # 索引有可能会大于列表长度，故需要截断处理，且每次的start都应该是上一个值的结束值
        start = end
        end = (start + n_way) % len(classes)

        if end < start:
            task_class = choose[start:] + choose[:end]
        else:
            task_class = choose[start: end]

        # 循环n_way次，取出k_shot个训练图像
        for i in task_class:
            for _ in range(k_shot):
                one_task.append(classes[i].pop(0))

        # 取出q_query个训练图像
        for i in task_class:
            for _ in range(q_query):
                one_task.append(classes[i].pop(0))

        dataset.append(one_task)

    return dataset

if __name__ == '__main__':
    image_classes = read_csv("./data/labels/train.csv")
    dataset = test(image_classes)

