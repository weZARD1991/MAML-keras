# -*- coding: utf-8 -*-
# @File : main.py
# @Author: Runist
# @Time : 2020/7/6 16:59
# @Software: PyCharm
# @Brief: 程序启动脚本

from dataReader import read_csv, task_split, data_generator
from net import MAMLmodel
from train import model_train
import tensorflow as tf
import config as cfg


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    image_list = read_csv("./data/labels/test.csv")
    task = task_split(image_list, support=cfg.num_support, query=cfg.num_querry, num_classes=cfg.num_classes)
    train_generator = data_generator(task, num_classes=cfg.num_classes)

    train_step = len(image_list) // cfg.batch_size

    maml_model = MAMLmodel(num_classes=cfg.num_classes)
    model_train(maml_model,
                cfg.epochs,
                train_generator,
                train_step=train_step,
                lr_outer=cfg.meta_lr,
                lr_inner=cfg.update_lr,
                num_classes=cfg.num_classes)


if __name__ == '__main__':
    main()
