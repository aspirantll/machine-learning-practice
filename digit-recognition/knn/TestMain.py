# -*- coding: utf-8 -*-
"""
@Time: 2018/6/24 0024
@Author: ll
@File: TestMain
@desc: 测试代码
"""
from knn.KnnModel import *
from sklearn import datasets
import pandas as pd
import numpy as np


def divide_ts(dataset):
    data_matrix = np.array([np.array((dataset.data[i], dataset.target[i])) for i in range(dataset.data.shape[0])])
    dataFrame = pd.DataFrame(data=data_matrix, columns=['data', 'target'])
    countSeries = dataFrame['target'].groupby(dataFrame['target']).count()
    exampleCount = pd.Series(data=(countSeries.values * 3)//4, index=countSeries.index)

    exampleData = []
    trainData = ([], [])
    for i in range(dataFrame.shape[0]):
        target = dataset.target[i]
        data = dataset.data[i]
        remainCount = exampleCount.get_value(target)
        if remainCount > 0:
            item = DataItem(data, target)
            exampleData.append(item)
            remainCount = remainCount - 1
            exampleCount.set_value(target, remainCount)
        else:
            trainData[0].append(data)
            trainData[1].append(target)

    return exampleData, (np.array(trainData[0]), np.array(trainData[1]))


if __name__=='__main__':
    digit_data = datasets.load_digits()
    exampleData, trainData = divide_ts(digit_data)
    for k in range(1,10):
        model = KnnModel(exampleData, k)
        result = model.classify(trainData[0])
        compare = result == trainData[1]
        trueCount = compare.sum()
        print('k-%d true:%d, false:%d , true rate: %f' % (k, trueCount, compare.shape[0]-trueCount, trueCount/compare.shape[0]))

