# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2020/7/7 10:06
# @Software: PyCharm
# @Brief: 数据读取脚本

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
import pandas as pd
import random
import numpy as np


def read_csv(csv_path):
    """
    读取包含图片名和标签的csv
    :param csv_path:
    :return:
    """
    csv = pd.read_csv(csv_path)
    image_label = []

    image_list = list("./data/images/" + csv.iloc[:, 0])

    label_list = pd.factorize(csv.iloc[:, 1])
    label_list = label_list[0].tolist()

    # for img, label in zip(image_list, label_list):
    #     image_label.append([img, label])

    # write_to_txt("task.txt", image_label)

    return image_list


def write_to_txt(file_name: str, contents):
    """
    将数据写入到txt
    :param file_name: 文件名
    :param contents: 内容
    :return: None
    """
    with open(file_name, encoding="utf-8", mode='w') as f:
        for c in contents:
            f.writelines(c)
            f.writelines('\n')


def task_split(image_label: list, support=9, query=1, step=600, num_classes=5):
    """
    将各个任务下的图片-标签，按任务分类。这个API是基于Mini-ImageNet下实现的，其中每个类只有600个
    故可以等间距选取，所以最终可以按照整数完整分配。
    :param image_label: {image: label}的字典
    :param support: support-set的数量
    :param query: query-set的数量
    :param step: 每个类中样本的数量
    :param num_classes: 一个任务分由几个类组成
    :return:
    """
    unit = support + query
    unit_num = step // unit     # unit下，一个类可以分成几个一组
    all_class = len(image_label) // step    # 总共有几类

    classes = [[] for _ in range(all_class)]
    dataset = []

    # 先将数据分成20种分类，每个分类里面以10个为一组（600张图就是60组），classes的shape => (20, 60)
    for i in range(all_class):
        for j in range(0, step, unit):
            start = i * step + j
            end = start + unit

            classes[i].append(image_label[start: end])

    # 循环60次，每次生成0-20的随机数，使得每个二级列表里的都长度都一致
    for _ in range(unit_num):
        choose = [i for i in range(len(classes))]
        random.shuffle(choose)

        # 循环20次，间距是5，每次取5个类，取出里面一组图片，作为一个任务
        for i in range(0, len(classes), num_classes):
            one_class = []

            # 循环五次，且是第一层循环里，5个随机数作为classes (20, 60)里的20的索引，60不需要索引，直接pop
            z = 0
            for j in choose[i: i+num_classes]:
                # 我们并不关心任务内的分类是对应所有分类的哪一个，所以并不需要原来的分类信息，只需要在任务内部再分类就好了
                new = ["{} {}".format(line, z) for line in classes[j].pop()]
                z += 1
                one_class += new
            dataset.append(one_class)

    return dataset


def process_one_task(one_task, num_classes=5):
    """
    对一个任务处理，对其中每一个图片进行读取
    :param one_task: 一个任务[img_path, label]
    :param num_classes: 一个任务中 分类的数量
    :return:
    """
    task = []
    for lines in one_task:
        img_path, label = lines.split()
        label = tf.one_hot(int(label), depth=num_classes)

        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        # 将unit8转为float32且归一化
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [224, 224])
        image = tf.expand_dims(image, axis=0)

        task.append([image, label])

    return task


def data_generator(all_task, num_classes=5):
    """
    数据生成器，利用生成器生成数据
    :param all_task: 存有全部任务的列表
    :param num_classes: 分类的数量
    :return:
    """
    n = len(all_task)
    i = 0
    while True:
        one_task = process_one_task(all_task[i], num_classes=num_classes)

        random.shuffle(one_task)
        i = (i + 1) % n

        yield one_task

