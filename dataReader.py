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


def task_split(image_label: list, q_query=1, one_class_img=600, n_way=5, k_shot=1):
    """
    将各个任务下的图片-标签，按任务分类。这个API是基于Mini-ImageNet下实现的，其中每个类只有600个
    故可以等间距选取，所以最终可以按照整数完整分配。n-way * k-shot张图片用来给inner loop训练，query是决定多少给out loop去test
    dataset最终是 , shape = [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
    :param image_label: {image: label}的字典
    :param q_query: query-set的数量，也就是在testing(out loop，为什么要叫test?)时，每个类别会有多少张照片，
    :param one_class_img: 每个类中样本的总共数量
    :param n_way: 一个任务由几个类组成
    :param k_shot: 每个类中有几张图片
    :return:
    """
    def split_train_and_test(one_task):
        """
        将one_class_list中按照k_shot和q_query的占比分成train和test
        :param one_task: 一个完整的任务
        :return:
        """
        train = []
        test = []
        for i in range(0, len(one_task), (k_shot + q_query)):
            train += one_task[i: i + k_shot]

        for i in range(0, len(one_task), (k_shot + q_query)):
            test += one_task[i: i + q_query]

        one_task = train + test
        return one_task

    unit = k_shot + q_query
    unit_num = one_class_img // unit     # unit下，一个类可以分成几个一组
    all_class = len(image_label) // one_class_img    # 总共有几类

    classes = [[] for _ in range(all_class)]
    dataset = []

    # 先将数据分成20种分类，每个分类里面以10个为一组（600张图就是60组），classes的shape => (20, 60)
    for i in range(all_class):
        for j in range(0, one_class_img, unit):
            start = i * one_class_img + j
            end = start + unit

            classes[i].append(image_label[start: end])

    # 循环60次，每次生成0-20的随机数，使得每个二级列表里在pop操作之后，长度都一致，就不会出现某一个类被取完了，其他类还没取完
    for _ in range(unit_num):
        choose = [i for i in range(len(classes))]
        random.shuffle(choose)

        # 循环20次，间距是5，每次取5个类，取出里面一组图片，作为一个任务
        for i in range(0, len(classes), n_way):
            one_class = []

            # 循环五次，且是第一层循环里，5个随机数作为classes (20, 60)里的20的索引，60不需要索引，直接pop
            # TODO: delete label information
            print(len(classes[i]))
            z = 0
            for j in choose[i: i+n_way]:
                # 我们并不关心任务内的分类是对应所有分类的哪一个，所以并不需要原来的分类信息，只需要在任务内部再分类就好了
                new = ["{} {}".format(line, z) for line in classes[j].pop()]
                z += 1
                one_class += new

            one_task = split_train_and_test(one_class)
            dataset.append(one_task)

    return dataset


def get_meta_batch(dataset, meta_batch_size):
    """
    生成一个batch的任务，用于训练。将传入的列表中的数据组合成一个batch_size
    :param dataset:
    :param meta_batch_size: batch_size个任务组成一个meta_batch
    :return: 生成一个batch的任务
    """
    while len(dataset) > 0:
        batch_task = []
        for i in range(meta_batch_size):
            try:
                data = process_one_task(dataset.pop(0))
                data = tf.squeeze(data, axis=1)
                batch_task.append(data)

            except IndexError:
                return

        # 将他们组合到新的任务里
        yield tf.stack(batch_task)


def get_batch_task(dataset, meta_batch_size):
    """
    从生成器中获取一个batch的任务
    :param dataset:
    :param meta_batch_size:
    :return:
    """
    try:
        one_task = next(batch_task_generate(dataset, meta_batch_size))
    except StopIteration:
        one_task = []
    return one_task


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


if __name__ == '__main__':
    image_list = read_csv("./data/labels/train.csv")
    dataset = task_split(image_list)

    # for step, batch_task in enumerate(get_meta_batch(dataset, 4)):
    #     print(batch_task.shape)
