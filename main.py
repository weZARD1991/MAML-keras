# -*- coding: utf-8 -*-
# @File : main.py
# @Author: Runist
# @Time : 2020/7/6 16:59
# @Software: PyCharm
# @Brief: 程序启动脚本

from dataReader import read_csv, make_datasets
from net import MAMLmodel
from train import model_train
import tensorflow as tf


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_image, train_list = read_csv("./data/labels/train.csv")
    train_dataset = make_datasets(train_image, train_list, batch_size=4, mode="train")
    train_step = len(train_image) // BATCH_SIZE

    maml_model = MAMLmodel(num_classes=64)
    model_train(maml_model, 1, train_dataset, train_step=train_step)


if __name__ == '__main__':
    BATCH_SIZE = 4

    main()
