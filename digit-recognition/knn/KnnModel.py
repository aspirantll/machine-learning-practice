# -*- coding: utf-8 -*-
"""
@Time: 2018/6/16 0016
@Author: ll
@File: KNNModel
@desc: knn算法模型
"""
from util import ReflectUtil
import pandas as pd
import numpy as np


class DataItem(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def data(self):
        return self.data

    def target(self):
        return self.target


class KnnModel(object):

    """
    K近邻算法模型
    """
    def __init__(self, data, k=20, similarity="euclidean"):
        """
        初始化
        :param data:样本图像数据集
        :param target:样本数据标签
        :param k:取最相似样本个数
        :param similarity:相似度计算方法
        """
        self.data = data
        self.k = k
        self.similarity = similarity
        self.getsimilarityfunc()

    def getsimilarityfunc(self):
        """
        初始化相似度计算函数
        :return:
        """
        if not hasattr(self, 'similarityfunc'):
            method_name = self.similarity
            if self.similarity.find('.')==-1:
                method_name = "knn.SimilarityFunc." + self.similarity
            self.similarityfunc = ReflectUtil.import_class(method_name)
        return self.similarityfunc

    def classify(self, data):
        result = []
        for item in data:
            similar_set = sorted(self.data, key=lambda c: self.getsimilarityfunc()(item, c.data))[:self.k]
            ser = pd.Series(item.target for item in similar_set)
            result.append(ser.mode().values[0])
        return np.array(result)

