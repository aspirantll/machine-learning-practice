# -*- coding: utf-8 -*-
"""
@Time: 2019/3/24 0024
@Author: ll
@File: test
@desc:
"""
from bayes import *
from util import *

if __name__=="__main__":
    posting_list = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','hime'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    class_vec = [0,1,0,1,0,1]

    test_list = [['love','my','dalmation'],['stupid','garbage']]

    bayes_model = Bayes(2, createVocabList(posting_list))
    bayes_model.train(posting_list, class_vec)
    print(bayes_model.predict(test_list))

