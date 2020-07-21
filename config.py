# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/7/8 16:54
# @Software: PyCharm
# @Brief: 配置文件

batch_size = 16
epochs = 40

inner_lr = 8e-6
outer_lr = 3e-9

n_way = 5
k_shot = 1
q_query = 1

width = 28
height = 28
channel = 1

save_path = "./logs/model/maml.h5"
