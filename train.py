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
               dataset,
               n_way=5,
               k_shot=1,
               q_query=1,
               lr_inner=0.001,
               lr_outer=0.002,
               batch_size=2,
               inner_train_step=1):
    """
    根据论文上Algorithm 1上的流程进行模型的训练
    :param model: MAML的模型
    :param epochs: 迭代轮次
    :param dataset: 数据集(无标签)
    :param lr_inner: 内层任务的学习率
    :param lr_outer: 外层任务的学习率
    :return:
    """
    outer_optimizer = optimizers.Adam(lr_outer)
    inner_optimizer = optimizers.Adam(lr_inner)
    train_step = len(dataset) // batch_size

    write_time = 0
    read_time = 0
    # Step 2：一个大循环
    for epoch in range(1, epochs + 1):
        i = 0

        for step, batch_task in enumerate(get_meta_batch(dataset, batch_size)):
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
                    # support_model（实质上是copy一份新的weights），是为了保证和上一次maml更新的模型方向一致
                    # 为了不破坏model的原来的weights，所以其实只有weights不一样
                    # support_model = copy_model(model, support_set)
                    t0 = time.time()
                    # model.save_weights("meta.h5")
                    meta_weights = model.get_weights()
                    t1 = time.time()
                    write_time += t1 - t0

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
                t2 = time.time()
                # model.load_weights("meta.h5")
                model.set_weights(meta_weights)
                t3 = time.time()
                read_time += t3 - t2

            gradients = query_tape.gradient(meta_batch_loss, model.trainable_variables)
            outer_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # 输出训练过程
            rate = round((i + 1) / train_step, 5)
            a = "*" * int(rate * 30)
            b = "." * int((1 - rate) * 30)
            print("\r{}/{} {:^3.0f}%[{}->{}] loss:{:.4f} accuracy:{:.4f}"
                  .format(step+1, train_step, int(rate * 100), a, b, meta_batch_loss, np.mean(task_acc)), end="")
            i += 1
        print()

    print("{} times save model using {}s, read_model using {}s.".format(train_step, write_time, read_time))

    return model


