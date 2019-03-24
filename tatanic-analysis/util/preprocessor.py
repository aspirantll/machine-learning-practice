# -*- coding: utf-8 -*-
"""
@Time: 2018/3/26 0026
@Author: ll
@File: preprocessing_util
@desc: 预处理工具
"""
from sklearn import preprocessing

# one-hot编码
def one_hot(data_set):
    enc = preprocessing.OneHotEncoder()
    return enc.fit_transform(data_set).toarray()





