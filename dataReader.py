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
import pandas as pd
import random
import numpy as np
import os
import config as cfg
import copy


def read_omniglot(path):
    classes = []

    for alphabet in os.listdir(path):
        # 语言路径
        alphabet_path = os.path.join(path, alphabet)

        for letter in os.listdir(alphabet_path):
            # 字体路径
            letter_path = os.path.join(alphabet_path, letter)
            letter_class = []
            # 具体图片路径
            for img_name in os.listdir(letter_path):
                img_path = os.path.join(letter_path, img_name)
                letter_class.append(os.path.normpath(img_path))
            classes.append(letter_class)

    rate = int(len(classes) * 0.8)
    train, valid = classes[:rate], classes[rate:]

    return train, valid


def read_miniimagenet(csv_path, one_class_img=600):
    """
    读取包含图片名和标签的csv
    :param csv_path:
    :param one_class_img: 一个类中有几张图片
    :return:
    """
    csv = pd.read_csv(csv_path)

    image_list = list("./data/miniImageNet/images/" + csv.iloc[:, 0])

    num_class = len(image_list) // one_class_img    # 总共有几类
    classes = [[] for _ in range(num_class)]

    # 先按照类区分开
    for i in range(num_class):
        start = i * one_class_img
        end = (i+1) * one_class_img
        classes[i] = image_list[start: end]

    return classes


def get_meta_batch(dataset, meta_batch_size, dataset_str):
    """
    生成一个batch的任务，用于训练。将传入的列表中的数据组合成一个batch_size
    :param dataset:
    :param meta_batch_size: batch_size个任务组成一个meta_batch
    :return: 生成一个batch的任务
    """
    if "dataset" not in get_meta_batch.__dict__:
        get_meta_batch.dataset = dataset_str

    if "index" not in get_meta_batch.__dict__ or get_meta_batch.dataset != dataset_str:
        get_meta_batch.index = 0
        get_meta_batch.dataset = dataset_str

    while True:
        batch_task = []
        get_meta_batch.index %= len(dataset)
        for i in range(meta_batch_size):
            try:
                data = process_one_task(dataset[get_meta_batch.index])
                data = tf.squeeze(data, axis=1)
                batch_task.append(data)

            except IndexError:
                return

            get_meta_batch.index += 1

        # 将他们组合到新的任务里
        yield tf.stack(batch_task)


def process_one_task(one_task, width=cfg.width, height=cfg.height):
    """
    对一个任务处理，对其中每一个图片进行读取
    :param one_task: 一个batch的任务[img_path, label]
    :param width:
    :param height:
    :return:
    """
    task = []

    for img_path in one_task:

        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        # 将unit8转为float32且归一化
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [width, height])

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
    classes = copy.deepcopy(classes)
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


def read_omniglot_ones(path, q_query=1, n_way=5, k_shot=1):
    """
    将各个分类下的img_path，按任务为单位分类。这个API是基于Mini-ImageNet下实现的，其中每个类只有600个
    为了均匀利用到所有数据，(q_query + k_shot) * n_way 要能被 图片数量整除
    。n-way * k-shot张图片用来给inner loop训练，n-way * query是给out loop去test
    dataset最终是 , shape = [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
    :param q_query: query-set的数量
    :param n_way: 一个任务由几个类组成
    :param k_shot: support-set数量
    :return:
    """
    file_list = []
    for alphabet in os.listdir(path):
        alphabet_path = os.path.join(path, alphabet)

        for letter in os.listdir(alphabet_path):
            # 字体路径
            letter_path = os.path.join(alphabet_path, letter)
            file_list.append(letter_path)

    image_list = []
    for img_path in file_list:
        sample = np.arange(20)
        np.random.shuffle(sample)

        img_list = os.listdir(img_path)
        img_list.sort()

        img_list = [os.path.join(img_path, img_list[i]) for i in sample[:q_query + k_shot]]
        image_list.append(img_list)

    sample = np.arange(len(image_list))
    np.random.shuffle(sample)

    dataset = []
    for start in range(0, len(sample), n_way):
        if len(sample[start: start + n_way]) < n_way:
            break

        train_task = []
        valid_task = []
        for i in sample[start: start + n_way]:
            train_task += image_list[i][:k_shot]
            valid_task += image_list[i][k_shot:]

        dataset.append(train_task + valid_task)

    return dataset[:640], dataset[640:]




