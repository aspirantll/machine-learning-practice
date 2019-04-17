# -*- coding: utf-8 -*-
"""
@Time: 2019/3/24 0024
@Author: ll
@File: bayes
@desc: 贝叶斯模型类
"""
from numpy import *

class Bayes(object):

    # class_categories表示分类个数
    def __init__(self, class_categories, vocab_list):
        # 分类数
        self.class_categories = class_categories
        # 词表
        self.vocab_list = vocab_list
        # 条件概率矩阵
        self.con_probabilitys = None
        # 类别概率
        self.cate_probabilitys = None

    # 训练方法
    def train(self, train_data, train_categories):
        vocab_vecs = self.setOfWords2Vec(train_data)
        self.bayesEstimate(vocab_vecs, train_categories)

    # 预测
    def predict(self, test_data):
        vocab_vecs = self.setOfWords2Vec(test_data)
        predict_hats = []

        for vocab_vec in  vocab_vecs:
            probs = [sum(array(vocab_vec)*self.con_probabilitys[i])+log(self.cate_probabilitys[i]) for i in range(self.class_categories)]
            predict_hats.append(probs.index(max(probs)))

        return predict_hats

    # 词表转词向量
    def setOfWords2Vec(self, data):
        vocab_vecs = []
        for datum in data:
            vocab_vec = [0]*len(self.vocab_list)
            for word in datum:
                if word in self.vocab_list:
                    vocab_vec[self.vocab_list.index(word)] = 1
                else:
                    print('word %s is not in vocab_list' % word)
            vocab_vecs.append(vocab_vec)
        return vocab_vecs

    # 贝叶斯估计
    def bayesEstimate(self, data, categories):
        """
        :param data: 训练样本数据向量
        :param categories: 样本数据类别
        :return:
        """
        # 先求类别概率
        cate_nums = zeros(self.class_categories)
        total = len(data) + self.class_categories
        for category in categories:
            cate_nums[category] = cate_nums[category]+1
        cate_probs = cate_nums/float(total)

        # 再计算某一类别条件下某词出现概率
        cate_words = [2]*self.class_categories
        cate_words_nums = ones((self.class_categories, len(self.vocab_list)))
        # 遍历数据集
        for i in range(len(data)):
            vocab_vec = data[i]
            cate = categories[i]
            cate_words[cate] = cate_words[cate]+sum(vocab_vec)
            cate_words_nums[cate] = cate_words_nums[cate] + vocab_vec

        # 计算概率
        con_probs = (cate_words_nums.T/cate_words).T

        self.cate_probabilitys = cate_probs
        self.con_probabilitys = con_probs