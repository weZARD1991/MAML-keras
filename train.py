# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2020/7/6 16:58
# @Software: PyCharm
# @Brief: 有关于训练，梯度优化的函数

from tensorflow.keras import optimizers, metrics
import tensorflow as tf
from net import *
from dataReader import get_meta_batch, create_label
import time


def maml_train(model,
               epochs,
               train_dataset,
               valid_dataset,
               n_way=5,
               k_shot=1,
               q_query=1,
               lr_inner=0.001,
               lr_outer=0.002,
               batch_size=2,
               inner_train_step=1):

    # Step 2：一个大循环
    for epoch in range(1, epochs + 1):
        step = 0
        train_step = len(train_dataset) // batch_size
        valid_step = len(valid_dataset) // batch_size

        # train
        for batch_id in range(train_step):
            batch_task = next(get_meta_batch(train_dataset, batch_size))
            loss, acc = maml_train_on_batch(model,
                                            batch_task,
                                            n_way=n_way,
                                            k_shot=k_shot,
                                            q_query=q_query,
                                            lr_inner=lr_inner,
                                            lr_outer=lr_outer,
                                            inner_train_step=inner_train_step)

            # 输出训练过程
            rate = (step + 1) / train_step
            a = "*" * int(rate * 30)
            b = "." * int((1 - rate) * 30)
            print("\r{}/{} {:^3.0f}%[{}->{}] loss:{:.4f} accuracy:{:.4f}"
                  .format(batch_id, train_step, int(rate * 100), a, b, loss, acc), end="")
            step += 1

        # valid
        for batch_id in range(valid_step):
            batch_task = next(get_meta_batch(valid_dataset, batch_size))
            loss, acc = maml_train_on_batch(model,
                                            batch_task,
                                            n_way=n_way,
                                            k_shot=k_shot,
                                            q_query=q_query,
                                            lr_inner=lr_inner,
                                            lr_outer=lr_outer,
                                            inner_train_step=inner_train_step)

            # 输出训练过程
            print("\r loss:{:.4f} accuracy:{:.4f}".format(loss, acc), end="")

        print()
    return model


def maml_train_on_batch(model,
                        batch_task,
                        n_way=5,
                        k_shot=1,
                        q_query=1,
                        lr_inner=0.001,
                        lr_outer=0.002,
                        inner_train_step=1,
                        is_train=True):
    """
    根据论文上Algorithm 1上的流程进行模型的训练
    :param model: MAML的模型
    :param lr_inner: 内层任务的学习率
    :param lr_outer: 外层任务的学习率
    :return:
    """
    outer_optimizer = optimizers.Adam(lr_outer)
    inner_optimizer = optimizers.Adam(lr_inner)

    # Step 3-4：采样一个batch的小样本任务，遍历生成的数据
    # 先生成一个batch的数据
    task_loss = []
    task_acc = []
    with tf.GradientTape() as query_tape:
        for one_task in batch_task:
            # Step 5：切分数据集为support set 和 query set
            support_set = one_task[:n_way * k_shot]
            query_set = one_task[n_way * k_shot:]

            # 直接进行前向传播，不然权重就是空的（前向传播不会改变权值）
            model.forward(support_set)
            # 读取出一份权重，再inner loop结束后再恢复回去.
            meta_weights = model.get_weights()

            train_label = create_label(n_way, k_shot)
            # Step 7：对support set进行梯度下降，求得meta-update的方向
            for inner_step in range(inner_train_step):
                with tf.GradientTape() as support_tape:
                    y_pred = model.forward(support_set)
                    support_loss = compute_loss(train_label, y_pred)

                gradients = support_tape.gradient(support_loss, model.trainable_variables)
                inner_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Step 6：评估一下模型
            valid_label = create_label(n_way, q_query)
            y_pred = model.forward(query_set)
            query_loss = compute_loss(valid_label, y_pred)

            equal_list = tf.equal(tf.argmax(y_pred, -1), tf.cast(valid_label, tf.int64))
            acc = tf.reduce_mean(tf.cast(equal_list, tf.float32))
            task_acc.append(acc)
            task_loss.append(query_loss)

        # Step 10：更新θ的权值，这里算的Loss是batch的loss平均
        meta_batch_loss = tf.reduce_mean(tf.stack(task_loss))

    model.set_weights(meta_weights)
    if is_train:
        gradients = query_tape.gradient(meta_batch_loss, model.trainable_variables)
        outer_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return meta_batch_loss, np.mean(task_acc)


def maml_eval(model,
              dataset,
              batch_size,
              n_way=5,
              k_shot=1,
              q_query=1,
              lr_inner=0.001,
              inner_train_step=3):

    inner_optimizer = optimizers.Adam(lr_inner)

    for batch_id, batch_task in enumerate(get_meta_batch(dataset, batch_size)):
        task_acc = []
        for one_task in batch_task:
            support_set = one_task[:n_way * k_shot]
            query_set = one_task[n_way * k_shot:]

            train_label = create_label(n_way, k_shot)
            # 测试的时候，正常梯度下降
            for inner_step in range(inner_train_step):
                with tf.GradientTape() as support_tape:
                    y_pred = model.forward(support_set)
                    support_loss = compute_loss(train_label, y_pred)

                gradients = support_tape.gradient(support_loss, model.trainable_variables)
                inner_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Step 6：评估一下模型
            valid_label = create_label(n_way, q_query)
            y_pred = model.forward(query_set)

            equal_list = tf.equal(tf.argmax(y_pred, -1), tf.cast(valid_label, tf.int64))
            acc = tf.reduce_mean(tf.cast(equal_list, tf.float32))
            task_acc.append(acc)

        print("Batch {} -- Testing accuracy: {:.4f}".format(batch_id, np.mean(task_acc)))
