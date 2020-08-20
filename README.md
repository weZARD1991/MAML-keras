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



## 三、MAML的核心算法

![Figure.png](https://i.loli.net/2020/08/20/JdoWSbLKe1xOfUr.png)

刚刚说了MAML关注的是，模型使用一份“适应性很强的”权重，它经过几次梯度下降就可以很好的适用于新的任务。那么我们训练的目标就变成了“如何找到这个权重”。而MAML作为其中一种实现方式，它先对一个batch中的每个任务都训练一遍，然后回到这个原始的位置，对这些任务的loss进行一个综合的判断，再选择一个适合所有任务的方向。

其中有监督学习的分类问题算法流程如下：

![Algorithm2.png](https://i.loli.net/2020/08/20/jg4N8u7JEePHrO1.png)

先决条件：

1. 以任务为单位的数据集
2. 两个学习率 $\alpha 、\beta$

流程解析（其中缩进代表在循环内部）：

Step 1: 随机初始化一个权重

Step 2: 一个while循环，对应的是训练中的epochs

​			Step 3: 采样一个batch的任务（假设为4个任务）

​			Step 4: for循环，用于遍历一个任务中的图片

​					Step 5: 从support set中取出一张图片和标签

​					Step 6-7: 对这一张图片进行前向传播，计算梯度后用$\alpha$反向传播，更新到$\theta'$中。

​					Step 8: 从query set中取出一张图片和标签进行meta-update

​			Step 10: 将所有用$\theta'$计算出来的损失求和，计算梯度后用$\beta$进行梯度下降，更新到$\theta$中

相关代码如下：

```Python
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
            model.set_weights(meta_weights)
            # Step 5：切分数据集为support set 和 query set
            support_x = one_task[:n_way * k_shot]
            query_x = one_task[n_way * k_shot:]
            support_y = create_label(n_way, k_shot)
            query_y = create_label(n_way, q_query)

            # Step 7-8：对support set进行梯度下降，求得meta-update的方向
            for inner_step in range(inner_train_step):
                with tf.GradientTape() as support_tape:
                    support_logits = model(support_x)
                    support_loss = compute_loss(support_y, support_logits)

                inner_grads = support_tape.gradient(support_loss, model.trainable_variables)
                inner_optimizer.apply_gradients(zip(inner_grads, model.trainable_variables))

            # 用query_set和θ’计算logits和loss
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
```

​					

## 四、MAML存在的问题与本代码的问题

MAML本身存在一些问题被发表在[How to train your MAML](https://arxiv.org/abs/1810.09502)中。而本代码在训练时也存在过拟合的问题，即训练初期，validation loss是有下降，但训练到后期，该validation loss逐步升高。而在调节了两个lr之后，该情况也不见好转。

