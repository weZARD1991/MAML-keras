# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/7/8 16:54
# @Software: PyCharm
# @Brief: 配置文件

batch_size = 4
eval_batch_size = 4
epochs = 40

inner_lr = 1e-2
outer_lr = 1e-3

n_way = 5
k_shot = 8
q_query = 1

width = 28
height = 28
channel = 1

save_path = "./logs/model/maml.h5"
log_dir = "./logs/summary/"

