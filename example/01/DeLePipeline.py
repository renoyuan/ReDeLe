#!/usr/bin/env python
#-*- coding: utf-8 -*-
#PROJECT_NAME: E:\project\ReDeLe
#CREATE_TIME: 2022-08-17 
#E_MAIL: renoyuan@foxmail.com
#AUTHOR: reno 

# 深度学习框架流程
# 1 数据处理 数据校验- 格式转化矩阵 重要
# 2 模型设计--网络结构 用别 人的结构
# 3 训练配置 算法优化 算力配置 改配置
# 4 训练 循环调用 向前计算 损失计算 反向传播 模板化
# 5 模型保存
import numpy as np
import json
import os
import sys
sys.path.insert(os.path.abspath(os.path.dirname(os.path.dirname((os.getcwd())))))
print(sys.path)
# 1.5 封装成load data函数

def load_data():
    # 1 读入训练数据
    datafile = './data/housing.data'
    data = np.fromfile(datafile, sep=' ')
    
    # 1.2 数据形状变换
    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    # # 1.3 数据集划分
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 1.4 数据归一化处理
    # 计算训练集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理 数据减去均值值再除以极差
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

# 获取数据 划分x,y
training_data, test_data = load_data()
x = training_data[:, :-1]
y = training_data[:, -1:]

# 查看数据
# print(x[0])
# print(y[0])


w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
w = np.array(w).reshape([13, 1]) # x,y

x1=x[0]

# print(x1)
# print(w)
t = np.dot(x1, w) # 内积


b = -0.2
z = t + b
# print(z)

# 将上述计算预测输出的过程以“类和对象”的方式来描述，类成员变量有参数www和bbb。通过写一个forward函数（代表“前向计算”）完成上述从特征和参数到输出预测值的计算过程，代码如下所示。
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        
        self.b = 0.
        print(self.w,self.b)

    # 2.2 模型设计 是深度学习模型关键要素之一，也称为网络结构设计，相当于模型的假设空间，即实现模型“前向计算”（从输入到输出）的过程。
    # 如果将输入特征和输出预测值均以向量表示，输入特征xxx有13个分量，yyy有1个分量，那么参数权重的形状（shape）是13×113\times113×1。假设我们以如下任意数字赋值参数做初始化：
    # 完整的线性回归公式，还需要初始化偏移量bbb，同样随意赋初值-0.2。那么，线性回归模型的完整输出是z=t+bz=t+bz=t+b，这个从特征和参数计算输出值的过程称为“前向计算”。
    # 从上述前向计算的过程可见，线性回归也可以表示成一种简单的神经网络（只有一个神经元，且激活函数为恒等式）。这也是机器学习模型普遍为深度学习模型替代的原因：由于深度学习网络强大的表示能力，很多传统机器学习模型的学习能力等同于相对简单的深度学习模型。
    def forward(self, x):
        # print("np.dot(x, self.w)",np.dot(x, self.w))
        z = np.dot(x, self.w) + self.b
        # print("z",z)
        return z
    
    # 2.3 训练配置 模型设计完成后，需要通过训练配置寻找模型的最优值，即通过损失函数来衡量模型的好坏。训练配置也是深度学习模型关键要素之一。
        # 如果要衡量预测放假和真实房价之间的差距，是否将每一个样本的差距的绝对值加和即可？差距绝对值加和是更加直观和朴素的思路，为何要平方加和？ 损失函数的设计不仅要考虑准确衡量问题的“合理性”，通常还要考虑“易于优化求解”。
    # 在回归问题中，均方误差是一种比较常见的形式，分类问题中通常会采用交叉熵作为损失函数，在后续的章节中会更详细的介绍。对一个样本计算损失函数值的实现如下。
    def loss(self, z, y):
        error = z - y
        # print("error",error)
        cost = error * error  # Loss = (y1 - z)*(y1 - z) 绝对值误差几何上是线性的不可微
        # print("cost",cost)
        cost = np.mean(cost)  # 因为计算损失函数时需要把每个样本的损失函数值都考虑到，所以我们需要对单个样本的损失函数进行求和，并除以样本总数NNN。 
        return cost
    
    # 梯度计算
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        
        return gradient_w, gradient_b
    
    #  更新梯度
    def update(self, gradient_w5, gradient_w9, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
    
    #  训练
    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if (i+1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses

    def predict(self,x,model):
        self.w,self.b=model
        z = self.forward(x)
        return z

# 使用定义的Network类，可以方便的计算预测值和损失函数。需要注意的是，类中的变量x, w，b, z, error等均是向量。以变量xxx为例，共有两个维度，一个代表特征数量（值为13），一个代表样本数量，代码如下所示。
# 2.3 训练配置
net = Network(13)
# 此处可以一次性计算多个样本的预测值和损失函数
x1 = x[0:3]
y1 = y[0:3]
z = net.forward(x1)
# print('predict: ', z)
loss = net.loss(z, y1)
# print('loss:', loss)

# 2.4 训练过程

# 2.4.1 梯度下降法
# 在现实中存在大量的函数正向求解容易，但反向求解较难，被称为单向函数，这种函数在密码学中有大量的应用。密码锁的特点是可以迅速判断一个密钥是否是正确的(已知xxx，求yyy很容易)，但是即使获取到密码锁系统，也无法破解出正确得密钥（已知yyy，求xxx很难）。
# 这种情况特别类似于一位想从山峰走到坡谷的盲人，他看不见坡谷在哪（无法逆向求解出LossLossLoss导数为0时的参数值），但可以伸脚探索身边的坡度（当前点的导数值，也称为梯度）。那么，求解Loss函数最小值可以这样实现：从当前的参数取值，一步步的按照下坡的方向下降，直到走到最低点。这种方法笔者称它为“盲人下坡法”。哦不，有个更正式的说法“梯度下降法”。

net = Network(13)
losses = []
#只画出参数w5和w9在区间[-160, 160]的曲线部分，以及包含损失函数的极值
w5 = np.arange(-160.0, 160.0, 1.0)
w9 = np.arange(-160.0, 160.0, 1.0)
losses = np.zeros([len(w5), len(w9)])

#计算设定区域内每个参数取值所对应的Loss
for i in range(len(w5)):
    for j in range(len(w9)):
        net.w[5] = w5[i]
        net.w[9] = w9[j]
        z = net.forward(x)
        loss = net.loss(z, y)
        losses[i, j] = loss

#使用matplotlib将两个变量和对应的Loss作3D图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)

w5, w9 = np.meshgrid(w5, w9)

ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap='rainbow')
# plt.show()
# plt.savefig("a.jpg")
"""
由此可见，均方误差表现的“圆滑”的坡度有两个好处：

曲线的最低点是可导的。
越接近最低点，曲线的坡度逐渐放缓，有助于通过当前的梯度来判断接近最低点的程度（是否逐渐减少步长，以免错过最低点）。
而绝对值误差是不具备这两个特性的，这也是损失函数的设计不仅仅要考虑“合理性”，还要追求“易解性”的原因。

现在我们要找出一组[w5,w9][w_5, w_9][w 5 ,w 9]的值，使得损失函数最小，实现梯度下降法的方案如下：

步骤1：随机的选一组初始值，例如：[w5,w9]=[−100.0,−100.0][w_5, w_9] = [-100.0, -100.0][w 5,w 9 ]=[−100.0,−100.0]
步骤2：选取下一个点[w5′,w9′][w_5^{'} , w_9^{'}][w 5′,w 9′]，使得L(w5′,w9′)<L(w5,w9)L(w_5^{'} , w_9^{'}) < L(w_5, w_9)L(w 5′,w 9′)<L(w 5 ,w 9 )
步骤3：重复步骤2，直到损失函数几乎不再下降。
如何选择[w5′,w9′][w_5^{'} , w_9^{'}][w 5′,w 9′]是至关重要的，第一要保证LLL是下降的，第二要使得下降的趋势尽可能的快。微积分的基础知识告诉我们：沿着梯度的反方向，是函数值下降最快的方向，如 图7 所示。简单理解，函数在某一个点的梯度方向是曲线斜率最大的方向，但梯度方向是向上的，所以下降最快的是梯度的反方向。
"""

# 2.4.2 梯度计算 可以通过具体的程序查看每个变量的数据和维度。
x1 = x[0]
y1 = y[0]
z1 = net.forward(x1)
print('x1 {}, shape {}'.format(x1, x1.shape))
print('y1 {}, shape {}'.format(y1, y1.shape))
print('z1 {}, shape {}'.format(z1, z1.shape))
gradient_w0 = (z1 - y1) * x1[0]
print('gradient_w0 {}'.format(gradient_w0))
x1 = x[0]
# 写一个for循环即可计算从w0w_0w 0到w12w_{12}w 12的所有权重的梯度
print(x,y)
for i in range(13):
   gradient=  (net.forward(x[0]) - y[0]) * x[0][i]
   print('gradient{} : {}'.format(i,gradient))


# 2.4.3 使用NumPy进行梯度计算 基于NumPy广播机制（对向量和矩阵计算如同对1个单一变量计算一样），可以更快速的实现梯度计算。计算梯度的代码中直接用(z1−y1)⋅x1(z_1 - y_1) \cdot x_1(z 1−y 1)⋅x 1​，得到的是一个13维的向量，每个分量分别代表该维度的梯度。
gradient_w = (z1 - y1) * x1
print('gradient_w_by_sample1 {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))
# 输入数据中有多个样本，每个样本都对梯度有贡献。如上代码计算了只有样本1时的梯度值，同样的计算方法也可以计算样本2和样本3对梯度的贡献。
# 注意这里是一次取出3个样本的数据，不是取出第3个样本
x3samples = x[0:3]
y3samples = y[0:3]
z3samples = net.forward(x3samples)

print('x {}, shape {}'.format(x3samples, x3samples.shape))
print('y {}, shape {}'.format(y3samples, y3samples.shape))
print('z {}, shape {}'.format(z3samples, z3samples.shape))
gradient_w = (z3samples - y3samples) * x3samples
print('gradient_w {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))

# 此处可见，计算梯度gradient_w的维度是3×133 \times 133×13，并且其第1行与上面第1个样本计算的梯度gradient_w_by_sample1一致，第2行与上面第2个样本计算的梯度gradient_w_by_sample2一致，第3行与上面第3个样本计算的梯度gradient_w_by_sample3一致。这里使用矩阵操作，可以更加方便的对3个样本分别计算各自对梯度的贡献。

# 小结一下这里使用NumPy库的广播功能：
# 一方面可以扩展参数的维度，代替for循环来计算1个样本对从w0w_0w 0到w12w_12w 12的所有参数的梯度。
# 另一方面可以扩展样本的维度，代替for循环来计算样本0到样本403对参数的梯度。
z = net.forward(x)
gradient_w = (z - y) * x
print('gradient_w shape {}'.format(gradient_w.shape))
print(gradient_w)

# 上面gradient_w的每一行代表了一个样本对梯度的贡献。根据梯度的计算公式，总梯度是对每个样本对梯度贡献的平均值。
# 可以使用NumPy的均值函数来完成此过程，代码实现如下。
# axis = 0 表示把每一行做相加然后再除以总的行数
gradient_w = np.mean(gradient_w, axis=0)
print('gradient_w ', gradient_w.shape)
print('w ', net.w.shape)
print(gradient_w)
print(net.w)
# 使用NumPy的矩阵操作方便地完成了gradient的计算，但引入了一个问题，gradient_w的形状是(13,)，而www的维度是(13, 1)。导致该问题的原因是使用np.mean函数时消除了第0维。为了加减乘除等计算方便，gradient_w和www必须保持一致的形状。因此我们将gradient_w的维度也设置为(13,1)，代码如下：

gradient_w = gradient_w[:, np.newaxis]
print('gradient_w shape', gradient_w.shape)

z = net.forward(x)
gradient_w = (z - y) * x
gradient_w = np.mean(gradient_w, axis=0)
gradient_w = gradient_w[:, np.newaxis]
gradient_w

# 述代码非常简洁地完成了www的梯度计算。同样，计算bbb的梯度的代码也是类似的原理。
gradient_b = (z - y)
gradient_b = np.mean(gradient_b)
# 此处b是一个数值，所以可以直接用np.mean得到一个标量


# 初始化网络
net = Network(13)
# 设置[w5, w9] = [-100., -100.]
net.w[5] = -100.0
net.w[9] = -100.0

z = net.forward(x)
loss = net.loss(z, y)
gradient_w, gradient_b = net.gradient(x, y)
gradient_w5 = gradient_w[5][0]
gradient_w9 = gradient_w[9][0]
print('point {}, loss {}'.format([net.w[5][0], net.w[9][0]], loss))
print('gradient {}'.format([gradient_w5, gradient_w9]))

# 梯度更新
# 下面研究更新梯度的方法，确定损失函数更小的点。首先沿着梯度的反方向移动一小步，找到下一个点P1，观察损失函数的变化。
# 在[w5, w9]平面上，沿着梯度的反方向移动到下一个点P1
# 定义移动步长 eta
eta = 0.1
# 更新参数w5和w9
net.w[5] = net.w[5] - eta * gradient_w5
net.w[9] = net.w[9] - eta * gradient_w9
# 重新计算z和loss
z = net.forward(x)
loss = net.loss(z, y)
gradient_w, gradient_b = net.gradient(x, y)
gradient_w5 = gradient_w[5][0]
gradient_w9 = gradient_w[9][0]
print('point {}, loss {}'.format([net.w[5][0], net.w[9][0]], loss))
print('gradient {}'.format([gradient_w5, gradient_w9]))
# 运行上面的代码，可以发现沿着梯度反方向走一小步，下一个点的损失函数的确减少了。感兴趣的话，大家可以尝试不停的点击上面的代码块，观察损失函数是否一直在变小。

# 相减：参数需要向梯度的反方向移动。
# eta：控制每次参数值沿着梯度反方向变动的大小，即每次移动的步长，又称为学习率。
# 为什么之前我们要做输入特征的归一化，保持尺度一致？这是为了让统一的步长更加合适，使训练更加高效。
# 特征输入归一化后，不同参数输出的Loss是一个比较规整的曲线，学习率可以设置成统一的值 ；特征输入未归一化时，不同特征对应的参数所需的步长不一致，尺度较大的参数需要大步长，尺寸较小的参数需要小步长，导致无法设置统一的学习率。
# https://ai-studio-static-online.cdn.bcebos.com/903f552bc55b4a5eba71caa7dd86fd2d7b71b8ebb6cb4500a5f5711f465707f3
# 2.4.5 封装Train函数

# 获取数据
train_data, test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]
# 创建网络
net = Network(13)
num_iterations=2000


# 画出损失函数的变化趋势
# plot_x = np.arange(num_iterations)
# plot_y = np.array(losses)
# plt.plot(plot_x, plot_y)
# plt.grid()
# f = plt.gcf()
# # plt.show()
# f.savefig(r'01.png')
# 2.4.6 训练过程扩展到全部参数
# 文演示的梯度下降的过程仅包含w5w_5w5和w9w_9w9两个参数。但房价预测的模型必须要对所有参数www和bbb进行求解，这需要将Network中的update和train函数进行修改。由于不再限定参与计算的参数（所有参数均参与计算），修改之后的代码反而更加简洁。

# 获取数据
train_data, test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]
# 创建网络
net = Network(13)
num_iterations=1000
# 启动训练
losses = net.train(x,y, iterations=num_iterations, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.grid()
f = plt.gcf()
# plt.show()
f.savefig(r'01.png')

# 2.4.7 随机梯度下降法（ Stochastic Gradient Descent）
# 在上述程序中，每次损失函数和梯度计算都是基于数据集中的全量数据。对于波士顿房价预测任务数据集而言，样本数比较少，只有404个。但在实际问题中，数据集往往非常大，如果每次都使用全量数据进行计算，效率非常低，通俗地说就是“杀鸡焉用牛刀”。由于参数每次只沿着梯度反方向更新一点点，因此方向并不需要那么精确。一个合理的解决方案是每次从总的数据集中随机抽取出小部分数据来代表整体，基于这部分数据计算梯度和损失来更新参数，这种方法被称作随机梯度下降法（Stochastic Gradient Descent，SGD），核心概念如下：

# mini-batch：每次迭代时抽取出来的一批数据被称为一个mini-batch。
# batch_size：一个mini-batch所包含的样本数目称为batch_size。
# epoch：当程序迭代的时候，按mini-batch逐渐抽取出样本，当把整个数据集都遍历到了的时候，则完成了一轮训练，也叫一个epoch。启动训练时，可以将训练的轮数num_epochs和batch_size作为参数传入

# 数据处理代码修改
# 数据处理需要实现拆分数据批次和样本乱序（为了实现随机抽样的效果）两个功能。
# 获取数据
train_data, test_data = load_data()
train_data.shape
# train_data中一共包含404条数据，如果batch_size=10，即取前0-9号样本作为第一个mini-batch，命名train_data1。
train_data1 = train_data[0:10]
train_data1.shape
# 使用train_data1的数据（0-9号样本）计算梯度并更新网络参数。
net = Network(13)
x = train_data1[:, :-1]
y = train_data1[:, -1:]
loss = net.train(x, y, iterations=1, eta=0.01)
loss
# 再取出10-19号样本作为第二个mini-batch，计算梯度并更新网络参数。
train_data2 = train_data[10:20]
x = train_data2[:, :-1]
y = train_data2[:, -1:]
loss = net.train(x, y, iterations=1, eta=0.01)
loss
# 按此方法不断的取出新的mini-batch，并逐渐更新网络参数。
# 接下来，将train_data分成大小为batch_size的多个mini_batch，如下代码所示：将train_data分成 40410+1=41\frac{404}{10} + 1 = 41 10404+1=41 个 mini_batch，其中前40个mini_batch，每个均含有10个样本，最后一个mini_batch只含有4个样本。
batch_size = 10
n = len(train_data)
mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]
print('total number of mini_batches is ', len(mini_batches))
print('first mini_batch shape ', mini_batches[0].shape)
print('last mini_batch shape ', mini_batches[-1].shape)
# 另外，这里是按顺序读取mini_batch，而SGD里面是随机抽取一部分样本代表总体。为了实现随机抽样的效果，我们先将train_data里面的样本顺序随机打乱，然后再抽取mini_batch。随机打乱样本顺序，需要用到np.random.shuffle函数，下面先介绍它的用法。

# 通过大量实验发现，模型受训练后期的影响更大，类似于人脑总是对近期发生的事情记忆的更加清晰。为了避免数据样本集合的顺序干扰模型的训练效果，需要进行样本乱序操作。当然，如果训练样本的顺序就是样本产生的顺序，而我们期望模型更重视近期产生的样本（预测样本会和近期的训练样本分布更接近），则不需要乱序这个步骤。

# 新建一个array
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
print('before shuffle', a)
np.random.shuffle(a)
print('after shuffle', a)

#2.5 模型保存
#Numpy提供了save接口，可直接将模型权重数组保存为.npy格式的文件。
np.save('w.npy', net.w)
np.save('b.npy', net.b)
