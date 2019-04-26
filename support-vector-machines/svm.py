# -*- coding: utf-8 -*-
"""
@Time: 2019/3/26 0026
@Author: ll
@File: svm
@desc: 线性支持向量机
"""

import numpy as np
from math import *


class SVM(object):
    # 初始化超参数
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel

    # 计算g(x)值
    def g(self, x, alpha, b, trainMat, trainLabels):
        n = trainMat.shape()[0]
        return np.sum([alpha[i]*trainLabels[i]*self.kernel(x,trainMat[i]) for i in range(n)]) + b


    # 训练数据
    def train(self, trainMat, trainLabels):
        n,m = trainMat.shape()
        # 设置初值
        alpha = np.zeros((n,1))
        b = np.sum(trainLabels)/n

        # 循环训练
        while True:
            # 计算g(x)值
            G = np.array([self.g(trainMat[i], alpha, b, trainMat, trainLabels) for i in range(n)])
            # 满足KKT条件状态
            dis = np.multiarray(G, trainLabels)
            status = [ 1 if alpha[i]==0 and  dis[i] < 1 or 0< alpha[i] < self.C and dis[i] != 1 or alpha[i]==self.C and dis[i] > 1 else 0 for i in range(n)]

            # 均满足KKT条件
            if 0 == sum(status):
                break

            # 阿尔法变量1
            i = status.index(1)

            # 计算E
            E = G - trainLabels
            delta = [fabs(E[j] - E[i]) for j in range(n)]

            # 阿尔法变量2
            j = delta.index(max(delta))

            # 更新无约束最优解
            eta = self.kernel(trainMat[i]-trainMat[j], trainMat[i]-trainMat[j])
            aiNew = alpha[i] - trainLabels[i]*(E[i]-E[j])/eta

            # 约束更新
            if trainLabels[i] == trainLabels[j]:
                L = max(0, alpha[i]+alpha[j]-self.C)
                H = min(self.C, alpha[i]-alpha[j])
            else:
                L = max(0, alpha[i]-alpha[j])
                H = min(self.C, self.C+alpha[i]-alpha[j])

            if aiNew > H:
                aiNew = H
            elif aiNew < L:
                aiNew = L

            ajNew = alpha[j] + trainLabels[i]*trainLabels[j]*(alpha[i]-aiNew)

            # 更新alpha,b
            alpha[i] = aiNew
            alpha[j] = ajNew

            if 0<aiNew<self.C:
                b = trainLabels[i] - G[i] + b
            elif 0<ajNew<self.C:
                b = trainLabels[j] - G[j] + b
            else:
                b = (trainLabels[i] - G[i] + trainLabels[j] - G[j])/2 + b
        self.alpha = alpha
        self.b = b
        self.trainMat = trainMat
        self.trainLabels = trainLabels

    # 预测
    def predict(self, x):
        if self.g(x, self.alpha, self.b, self.trainMat, self.trainLabels) > 0:
            return 1
        else:
            return 0


