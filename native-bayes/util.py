# -*- coding: utf-8 -*-
"""
@Time: 2019/3/24 0024
@Author: ll
@File: util
@desc: 工具类
"""


# 根据训练数据，得出词表
def createVocabList(data):
    vocab_set = set([])
    for datum in data:
        vocab_set = vocab_set | set(datum)
    return list(vocab_set)




