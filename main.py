# -*- coding: utf-8 -*-
# @File : main.py
# @Author: Runist
# @Time : 2020/7/6 16:59
# @Software: PyCharm
# @Brief: 程序启动脚本

from dataReader import *
from train import *
from net import *
import tensorflow as tf
import config as cfg
import os
from tqdm import tqdm


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # maml_model = MAMLmodel(num_classes=cfg.n_way)
    maml_model = MAML_model(num_classes=cfg.n_way)
    # Step 2：一个大循环
    for epoch in range(1, cfg.epochs + 1):
        # 把数据读取放入Epoch里面，每次读出来的任务里面图片组合都不同
        train_list, valid_list = read_omniglot("./data/omniglot/images_background")
        train_dataset = task_split(train_list, q_query=cfg.q_query, n_way=cfg.n_way, k_shot=cfg.k_shot)
        valid_dataset = task_split(valid_list, q_query=cfg.q_query, n_way=cfg.n_way, k_shot=cfg.k_shot)

        train_step = len(train_dataset) // cfg.batch_size
        valid_step = len(valid_dataset) // cfg.batch_size

        # train
        process_bar = tqdm(range(train_step), ncols=100, desc="Epoch {}".format(epoch), unit="step")
        for _ in process_bar:
            batch_task = next(get_meta_batch(train_dataset, cfg.batch_size))
            loss, acc = maml_train_on_batch(maml_model,
                                            batch_task,
                                            n_way=cfg.n_way,
                                            k_shot=cfg.k_shot,
                                            q_query=cfg.q_query,
                                            lr_inner=cfg.inner_lr,
                                            lr_outer=cfg.outer_lr,
                                            inner_train_step=1)

            process_bar.set_postfix({'loss': '{:.5f}'.format(loss),
                                     'acc': '{:.5f}'.format(acc)})

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

        # 输出验证结果
        print("val_loss:{:.4f} val_accuracy:{:.4f}\n".format(np.mean(val_loss), np.mean(val_acc)))

    # test
    # test_list = read_csv("./data/labels/test.csv")
    # test_dataset = task_split(test_list)
    # maml_eval(maml_model, test_dataset, lr_outer=cfg.inner_lr, lr_inner=cfg.outer_lr, batch_size=cfg.batch_size)
    maml_model.save_weights(cfg.save_path)


if __name__ == '__main__':
    main()
