# -*- coding: utf-8 -*-
"""
@Time: 2018/6/20 0020
@Author: ll
@File: SimilarityFunc
@desc:相似度计算方法
"""
import numpy as np

def euclidean(one, another):
    npOne = np.array(one)
    npAnnother = np.array(another)
    return np.sqrt(np.sum(np.square(npOne-npAnnother)))
