# Model-Agnostic Meta-Learning  - MAML

## 一、相关概念：

### 1、meta-leaning

meta-leaning指的是元学习，元学习是深度学习的一个分支，一个好的元模型（meta-learner）应该具备对新的、少量的数据做出快速而准确的学习。通俗的来讲， 对于人而言，你会玩CS就很容易上手CF。但是对于神经网络来说，并非如此。如果让一个汽车分类网络去识别不同的单车，那效果肯定很差。而传统的CNN网络都是输入大量的数据，然后进行分类的学习。但是这样做的问题就是，神经网络的通用性太差了，根本达不到“智能”的标准。人类的认知系统，可以通过少量的数据就可以从中学习到规律。

### 2、few-shot learning

few-shot learning译为小样本学习，是指从极少的样本中学习出一个模型。

**N-way K-shot** 

这是小样本学习中常用的数据，用以描述一个任务：它包含N个分类，每个分类只有K张图片。

**Support set and Query set**

Support set指的是参考集，Query set指的是测试集。用人识别动物种类大比分，有5种不同的动物，每种动物2张图片，这样10张图片给人做参考。另外给出5张动物图片，让人去判断各自属于那一种类。那么10张作为参考的图片就称为Support set，5张测试图片就称为Query set。

<img src="https://i.loli.net/2020/07/31/jKm4AituQxqZaTP.png" alt="image-20200731165740436.png" style="zoom:50%;" />

## 二、什么是MAML?

[论文地址](https://arxiv.org/pdf/1703.03400.pdf)

### 1、要解决的问题 

- 小样本问题
- 模型收敛太慢

普通的分类、检测任务中，因为分类、检测物体的类别是已知的，可以收集大量数据来训练。例如 VOC、COCO 等检测数据集，都有着上万张图片用于训练。而如果我们仅仅只有几张图片用于训练，这给模型预测带来很大障碍。

在深度学习中，解决训练数据不足常用的一个技巧是“预训练-微调”（Pretraining-finetune），即大数据集上面预训练模型，然后在小数据集上去微调权重。但是，在训练数据极其稀少的时候（仅有个位数的训练图片），这个技巧是无法奏效的。并且这样的方式有时候反而会让模型陷入局部最优。

### 2、MAML的关键点

本文的设想是训练**一组初始化参数**，模型通过初始化参数，仅用少量数据就能实现快速收敛的效果。为了达到这一目的，模型需要大量的**先验知识**来不停修正初始化参数，使其能够适应不同种类的数据。

### 3、MAML与Pretraining的区别

- Pretraining

假设有一个模型从task1的数据中训练出来了一组权重，我们记为$\theta1$，这个$\theta1$是图中深绿色的点，可以看到，在task1下，他已经达到了全局最优。而如果我们的模型如果用$\theta1$作为task2的初始值，我们最终将会到达浅绿色的点，而这个点只是task2的局部最优点。产生这样的问题也很简单，就是因为模型在训练task1的数据时并不用考虑task2的数据。

<img src="https://i.loli.net/2020/07/31/qPtvoZ9FpdWuHKE.png" alt="pretraining.png" style="zoom:50%;" />

- MAML

MAML则需要同时考虑两个数据集的分布，假设MAML经过训练以后得到了一组权重我们记为$\theta2$，虽然从图中来看，这个权重对于两个任务来说，都没有达到全局最优。但是很明显，**经过训练以后**，他们都能收敛到全局最优。

<img src="https://i.loli.net/2020/07/31/Igp9UNjs7o2mZhE.png" alt="maml.png" style="zoom:50%;" />

所以，Pretraining每次强调的都是**当下**这个模型能不能达到最优，而MAML强调的则是**经过训练**以后能不能达到最优。



