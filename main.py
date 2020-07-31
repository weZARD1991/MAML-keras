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
import PIL


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    maml_model = MAML_model(num_classes=cfg.n_way)
    # maml_model.build(input_shape=(None, 28, 28, 1))
    # maml_model.load_weights(cfg.save_path)
    # 直接进行前向传播，不然权重就是空的（前向传播不会改变权值），如果是用keras的Sequential来建立模型就自动初始化了

    # 把数据读取放入Epoch里面，每次读出来的任务里面图片组合都不同
    # train_list = read_miniimagenet("./data/miniImageNet/labels/train.csv")
    # valid_list = read_miniimagenet("./data/miniImageNet/labels/val.csv")
    train_list, valid_list = read_omniglot("./data/omniglot/images_background")
    # train_list, valid_list = read_omniglot_ones("./data/enhance_omniglot/Omniglot/images_background")
    train_dataset = task_split(train_list, q_query=cfg.q_query, n_way=cfg.n_way, k_shot=cfg.k_shot)
    valid_dataset = task_split(valid_list, q_query=cfg.q_query, n_way=cfg.n_way, k_shot=cfg.k_shot)

    train_iter = DataIter(train_dataset)
    valid_iter = DataIter(valid_dataset)

    # 删除上次训练留下的summary文件
    if not os.path.exists(cfg.log_dir):
        os.mkdir(cfg.log_dir)
    for file in os.listdir(cfg.log_dir):
        os.remove(os.path.join(cfg.log_dir, file))

    # 创建summary
    summary_writer = tf.summary.create_file_writer(logdir=cfg.log_dir)

    for epoch in range(1, cfg.epochs + 1):

        train_step = len(train_dataset) // cfg.batch_size
        valid_step = len(valid_dataset) // cfg.eval_batch_size

        train_loss = []
        train_acc = []

        # train
        process_bar = tqdm(range(train_step), ncols=100, desc="Epoch {}".format(epoch), unit="step")
        for _ in process_bar:
            batch_task = get_meta_batch(train_iter, cfg.batch_size)
            loss, acc = maml_train_on_batch(maml_model,
                                            batch_task,
                                            n_way=cfg.n_way,
                                            k_shot=cfg.k_shot,
                                            q_query=cfg.q_query,
                                            lr_inner=cfg.inner_lr,
                                            lr_outer=cfg.outer_lr,
                                            inner_train_step=1)

            train_loss.append(loss)
            train_acc.append(acc)
            process_bar.set_postfix({'loss': '{:.5f}'.format(loss), 'acc': '{:.5f}'.format(acc)})
        
        # 输出平均后的训练结果
        print("\rtrain_loss:{:.4f} train_accuracy:{:.4f}".format(np.mean(train_loss), np.mean(train_acc)))

        val_acc = []
        val_loss = []

        # valid
        process_bar = tqdm(range(valid_step), ncols=100, desc="Epoch {}".format(epoch), unit="step")
        for _ in process_bar:
            batch_task = get_meta_batch(valid_iter, cfg.eval_batch_size)
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

            process_bar.set_postfix({'val_loss': '{:.5f}'.format(loss), 'val_acc': '{:.5f}'.format(acc)})

        # 输出平均后的验证结果
        print("\rvalidation_loss:{:.4f} validation_accuracy:{:.4f}\n".format(np.mean(val_loss), np.mean(val_acc)))

        # 保存到tensorboard里
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', np.mean(train_loss), step=epoch)
            tf.summary.scalar('train_acc', np.mean(train_acc), step=epoch)
            tf.summary.scalar('valid_loss', np.mean(val_loss), step=epoch)
            tf.summary.scalar('valid_acc', np.mean(val_acc), step=epoch)

    # test
    # test_list = read_csv("./data/labels/test.csv")
    # test_dataset = task_split(test_list)
    # maml_eval(maml_model, test_dataset, lr_outer=cfg.inner_lr, lr_inner=cfg.outer_lr, batch_size=cfg.batch_size)
    maml_model.save_weights(cfg.save_path)


if __name__ == '__main__':
    main()
