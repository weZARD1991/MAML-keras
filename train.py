# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2020/7/6 16:58
# @Software: PyCharm
# @Brief: 有关于训练，梯度优化的函数

from tensorflow.keras import optimizers, metrics
import time

from dataReader import generate_dataset, SinusoidGenerator
from net import *

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random


def model_train(model, epochs, dataset, train_step, lr_inner=0.01, lr_outer=0.002, update_times=2):
    """
    根据论文上Algorithm 1上的流程进行模型的训练
    :param model:
    :param epochs:
    :param train_step:
    :param dataset:
    :param lr_inner:
    :param lr_outer:
    :param update_times:
    :return:
    """
    optimizer = optimizers.Adam(lr_outer)
    maml_loss = metrics.Mean(name='maml_loss')

    # Step 2：一个大循环
    for epoch in range(1, epochs + 1):
        maml_loss.reset_states()
        step = 0

        # Step 3：采样一个batch的小样本任务（我们通过外部传入的dataset采样）
        # Step 4：遍历生成的数据
        for batch_id, (images, y_true) in enumerate(dataset.take(train_step)):
            model.forward(images)  # 直接进行前向传播，以初始化模型

            with tf.GradientTape() as test_tape:
                test_tape.watch(model.trainable_variables)

                for _ in range(update_times):
                    # Step 5：根据K个例子评估模型
                    with tf.GradientTape() as train_tape:
                        y_pred = model.forward(images)
                        train_loss = compute_loss(y_true, y_pred)

                    # Step 6：计算梯度，更新参数
                    gradients = train_tape.gradient(train_loss, model.trainable_variables)
                    # copy过后的model（实质上是在不同task上的模型），是为了保证和上一次maml更新的模型方向一致
                    # 这个copy的模型每次都是不一样的，用来更新
                    task_model = copy_model(model, MAMLmodel(num_classes=64), images)

                    # 更新copy的模型参数
                    k = 0
                    for j in range(len(task_model.layers)):
                        if "max_pooling2d" in task_model.layers[j].name or "flatten" in task_model.layers[j].name:
                            continue
                        task_model.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                                                                           tf.multiply(lr_inner, gradients[k]))
                        task_model.layers[j].bias = tf.subtract(model.layers[j].bias,
                                                                         tf.multiply(lr_inner, gradients[k + 1]))
                        k += 2

                # Step 8：计算在copy的模型上的loss
                y_pred = task_model.forward(images)
                test_loss = compute_loss(y_true, y_pred)

            # Step 8：计算在task上的梯度，根据梯度确定方向，更新maml模型参数
            gradients = test_tape.gradient(test_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # 更新maml_loss
            maml_loss.update_state(test_loss)
            # 输出训练过程
            rate = (step + 1) / train_step
            a = "*" * int(rate * 30)
            b = "." * int((1 - rate) * 30)
            loss = maml_loss.result().numpy()
            print("\r{}/{} {:^3.0f}%[{}->{}] loss:{:.4f}"
                  .format(batch_id, train_step, int(rate * 100), a, b, loss), end="")
            step += 1
        print()

    return model


