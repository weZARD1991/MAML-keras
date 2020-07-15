# -*- coding: utf-8 -*-
# @File : main.py
# @Author: Runist
# @Time : 2020/7/6 16:59
# @Software: PyCharm
# @Brief: 程序启动脚本

from dataReader import read_csv, task_split
from net import MAMLmodel
from train import maml_train, maml_eval
import tensorflow as tf
import config as cfg


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_list = read_csv("./data/labels/train.csv")
    test_list = read_csv("./data/labels/test.csv")
    train_dataset = task_split(train_list)
    test_dataset = task_split(test_list)

    maml_model = MAMLmodel(num_classes=cfg.num_classes)
    maml_model = maml_train(maml_model,
                             cfg.epochs,
                             train_dataset,
                             n_way=cfg.n_way,
                             k_shot=cfg.k_shot,
                             q_query=cfg.q_query,
                             lr_outer=cfg.meta_lr,
                             lr_inner=cfg.update_lr,
                             batch_size=cfg.batch_size)

    model_eval(maml_model, test_dataset, batch_size=cfg.batch_size)
    # maml_model.save_weights(cfg.save_path)


if __name__ == '__main__':
    main()
