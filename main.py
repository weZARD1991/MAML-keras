# -*- coding: utf-8 -*-
# @File : main.py
# @Author: Runist
# @Time : 2020/7/6 16:59
# @Software: PyCharm
# @Brief: 程序启动脚本

from dataReader import *
from train import *
from net import MAMLmodel
import tensorflow as tf
import config as cfg
import os


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    maml_model = MAMLmodel(num_classes=cfg.n_way)

    # Step 2：一个大循环
    for epoch in range(1, cfg.epochs + 1):
        # 把数据读取放入Epoch里面，每次读出来的任务里面图片组合都不同
        train_list = read_csv("./data/labels/train.csv")
        valid_list = read_csv("./data/labels/val.csv")
        train_dataset = task_split(train_list)
        valid_dataset = task_split(valid_list)

        step = 0
        train_step = len(train_dataset) // cfg.batch_size
        valid_step = len(valid_dataset) // cfg.batch_size

        # train
        for batch_id in range(train_step):
            batch_task = next(get_meta_batch(train_dataset, cfg.batch_size))
            loss, acc = maml_train_on_batch(maml_model,
                                            batch_task,
                                            n_way=cfg.n_way,
                                            k_shot=cfg.k_shot,
                                            q_query=cfg.q_query,
                                            lr_inner=cfg.inner_lr,
                                            lr_outer=cfg.outer_lr,
                                            inner_train_step=1)

            # 输出训练过程
            rate = (step+1) / train_step
            a = "*" * int(rate * 30)
            b = "." * int((1 - rate) * 30)
            print("\r{}/{} {:^3.0f}%[{}->{}] loss:{:.4f} accuracy:{:.4f}"
                  .format(batch_id + 1, train_step, int(rate * 100), a, b, loss, acc), end="")
            step += 1
        print()

        val_acc = []
        val_loss = []
        # valid
        for batch_id in range(valid_step):
            batch_task = next(get_meta_batch(valid_dataset, cfg.batch_size))
            loss, acc = maml_train_on_batch(maml_model,
                                            batch_task,
                                            n_way=cfg.n_way,
                                            k_shot=cfg.k_shot,
                                            q_query=cfg.q_query,
                                            lr_inner=cfg.inner_lr,
                                            lr_outer=cfg.outer_lr,
                                            inner_train_step=3,
                                            meta_update=False)
            val_loss.append(loss)
            val_acc.append(acc)
        # 输出训练过程
        print("val_loss:{:.4f} val_accuracy:{:.4f}".format(np.mean(val_loss), np.mean(val_acc)))

    # test
    test_list = read_csv("./data/labels/test.csv")
    test_dataset = task_split(test_list)
    # maml_eval(maml_model, test_dataset, lr_outer=cfg.inner_lr, lr_inner=cfg.outer_lr, batch_size=cfg.batch_size)
    maml_model.save_weights(cfg.save_path)


if __name__ == '__main__':
    main()
