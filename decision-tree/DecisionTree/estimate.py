# -*- coding: utf-8 -*-
"""
@Time: 2018/6/29 0029
@Author: ll
@File: estimate
@desc: 属性评估算法
:param data and column
:return split_data score
"""

import pandas as pd
import math


def information_gain(data, column):
    """
    计算信息增益
    :param data: 数据集
    :param column: 所选列
    :return: 按属性划分数据 增益率
    """
    ent = __ent(data)
    split_data = {}
    total_count = data.count().values[0]
    for value, group in data.groupby(data[column]):
        split_data[value] = group
        ent = ent - group.count().values[0]*__ent(group)/total_count
    return split_data, ent


def __ent(data, label_column='label'):
    """
    信息熵计算
    :param data: 数据集
    :param label_column: 标签列
    :return: 信息熵
    """
    label_counter = pd.value_counts(data[label_column])
    total_count = data.count().values[0]
    ent = 0
    for count in label_counter:
        pk = count / total_count
        ent = ent - pk*math.log2(pk)
    return ent
