# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/7/8 16:54
# @Software: PyCharm
# @Brief: 配置文件

batch_size = 4
epochs = 5

inner_lr = 1e-6
outer_lr = 1e-8

n_way = 5
k_shot = 1
q_query = 1

update_times = 2

save_path = "./logs/model/maml.h5"
