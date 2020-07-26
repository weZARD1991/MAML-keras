# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2020/7/6 16:58
# @Software: PyCharm
# @Brief: 有关于训练，梯度优化的函数

from tensorflow.keras import optimizers, losses, metrics
import tensorflow as tf
from dataReader import get_meta_batch, create_label
import numpy as np


def compute_loss(y_true, y_pred):
    """
    计算loss
    :param y_true: 模型
    :param y_pred:
    :return:
    """
    mse = losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

    return mse


def maml_eval(model,
              test_dataset,
              n_way=5,
              k_shot=1,
              q_query=1,
              lr_inner=0.001,
              lr_outer=0.002,
              batch_size=2):
    """
    maml的测试函数
    :param model: 经过训练后的模型
    :param test_dataset: 测试集
    :param n_way: 一个任务内分类的数量
    :param k_shot: support set
    :param q_query: query set
    :param lr_inner: 内层support set的学习率
    :param lr_outer: 外层query set任务的学习率
    :param batch_size:
    :return: None
    """
    test_step = len(test_dataset) // batch_size

    test_acc = []
    test_loss = []

    for batch_id in range(test_step):
        batch_task = next(get_meta_batch(test_dataset, batch_size))
        loss, acc = maml_train_on_batch(model,
                                        batch_task,
                                        n_way=n_way,
                                        k_shot=k_shot,
                                        q_query=q_query,
                                        lr_inner=lr_inner,
                                        lr_outer=lr_outer,
                                        inner_train_step=5,
                                        meta_update=False)
        test_acc.append(loss)
        test_loss.append(acc)

        # 输出测试结果
        print("test_loss:{:.4f} test_accuracy:{:.4f}".format(np.mean(test_acc), np.mean(test_loss)))


def maml_train_on_batch(model,
                        batch_task,
                        n_way=5,
                        k_shot=1,
                        q_query=1,
                        lr_inner=0.001,
                        lr_outer=0.002,
                        inner_train_step=1,
                        meta_update=True):
    """
    根据论文上Algorithm 1上的流程进行模型的训练
    :param model: MAML的模型
    :param batch_task: 一个batch 的任务
    :param n_way: 一个任务内分类数量
    :param k_shot: support set的数量
    :param q_query: query的数量
    :param lr_inner: 内层support set的学习率
    :param lr_outer: 外层query set任务的学习率
    :param inner_train_step: 内层support set的训练次数
    :param meta_update: 是否进行meta update
    :return: loss, accuracy -- 都是均值
    """
    outer_optimizer = optimizers.Adam(lr_outer)
    inner_optimizer = optimizers.Adam(lr_inner)

    # Step 3-4：采样一个batch的小样本任务，遍历生成的数据
    # 先生成一个batch的数据
    task_loss = []
    task_acc = []

    # 读取出一份权重，在update一个batch的任务之后再恢复回去
    meta_weights = model.get_weights()

    with tf.GradientTape() as query_tape:
        for one_task in batch_task:
            # Step 5：切分数据集为support set 和 query set
            support_x = one_task[:n_way * k_shot]
            query_x = one_task[n_way * k_shot:]
            support_y = create_label(n_way, k_shot)
            query_y = create_label(n_way, q_query)

            # with tf.GradientTape() as support_tape:
            #     support_logits = model(support_x)
            #     support_loss = compute_loss(support_y, support_logits)
            #
            # inner_grads = support_tape.gradient(support_loss, model.trainable_variables)
            # inner_optimizer.apply_gradients(zip(inner_grads, model.trainable_variables))
            #
            # query_logits = model(query_x)
            # query_loss = compute_loss(query_y, query_logits)
            # task_loss.append(query_loss)

            # Step 7-8：对support set进行梯度下降，求得meta-update的方向
            for inner_step in range(inner_train_step):
                with tf.GradientTape() as support_tape:
                    support_logits = model(support_x)
                    support_loss = compute_loss(support_y, support_logits)

                inner_grads = support_tape.gradient(support_loss, model.trainable_variables)
                inner_optimizer.apply_gradients(zip(inner_grads, model.trainable_variables))

            # Step 6：评估一下模型
            query_logits = model(query_x)
            query_pred = tf.nn.softmax(query_logits)
            query_loss = compute_loss(query_y, query_logits)

            equal_list = tf.equal(tf.argmax(query_pred, -1), tf.cast(query_y, tf.int64))
            acc = tf.reduce_mean(tf.cast(equal_list, tf.float32))
            task_acc.append(acc)
            task_loss.append(query_loss)

        # Step 10：更新θ的权值，这里算的Loss是batch的loss平均
        meta_batch_loss = tf.reduce_mean(tf.stack(task_loss))
        model.set_weights(meta_weights)

    if meta_update:
        outer_grads = query_tape.gradient(meta_batch_loss, model.trainable_variables)
        outer_optimizer.apply_gradients(zip(outer_grads, model.trainable_variables))

    return meta_batch_loss, np.mean(task_acc)


