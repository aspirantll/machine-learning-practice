# -*- coding: utf-8 -*-
"""
@Time: 2018/3/26 0026
@Author: ll
@File: pandas_util
@desc: pandas工具类
"""
import pandas as pd
import logging
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)



#读取csv文件
def read_csv(file_path):
    logging.info("read in %s" % file_path)
    return pd.read_csv(file_path)


# Series.value_counts
def value_counts(data_set):
    value_count_dict = data_set.value_counts()
    logging.info("value_counts:\n%s" % value_count_dict)
    return value_count_dict

