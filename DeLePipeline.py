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
# 读入训练数据
datafile = './work/housing.data'
data = np.fromfile(datafile, sep=' ')