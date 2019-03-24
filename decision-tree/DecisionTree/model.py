# -*- coding: utf-8 -*-
"""
@Time: 2018/6/28 0028
@Author: ll
@File: model
@desc: 决策树模型类
"""

from util import ReflectUtil
import pandas as pd

class DTNode(object):
    """
    决策树节点
    """
    def __init__(self):
       pass

    def set_column(self, column):
        """
        设置选择列
        :param column:
        :return:
        """
        self.column = column

    def set_target(self, target):
        """
        叶节点设置目标类型
        :param target: 目标分类
        :return:
        """
        self.target = target

    def add_child(self, value, node):
        """
        添加属性值分支
        :param value:
        :param node:
        :return:
        """
        if not hasattr(self, 'children'):
            self.children = {}
        self.children[value] = node
        if not hasattr(self, 'default'):
            self.default = node

    def get_target(self):
        """
        获取目标分类标签(叶节点)
        :return:
        """
        if not hasattr(self, 'target'):
            return None
        return self.target

    def get_column(self):
        """
        获取被划分列(非叶子结点)
        :return:
        """
        if not hasattr(self, 'column'):
            return None
        return self.column

    def get_child(self, value):
        """
        根据属性值获取分支
        :param value:
        :return:
        """
        if not hasattr(self, 'children'):
            return None
        return self.children.get(value, self.default)



class DTModel(object):
    """
    决策树模型：基于pandas数据结构
    """

    def __init__(self):
        self.max_level = 5
        self.estimate = None
        self.model_tree = None
        self.LABELS = 'label'

    def __set_estimate(self, estimate):
        """
        初始化属性评估函数
        :return:
        """
        method_name = estimate
        if estimate.find('.') == -1:
            method_name = "DecisionTree.estimate." + estimate
        self.estimate = ReflectUtil.import_class(method_name)

    def fit(self, data_frame, max_level=5, estimate='information_gain'):
        """
        构造简单决策树模型
        :param max_level: 最大属性选择数目
        :param estimate: 评估算法
        :param data_frame: 训练数据，labels列作为标签列，不可缺少
        """
        self.max_level = max_level
        self.__set_estimate(estimate)

        # 所有属性
        columns = data_frame.columns.tolist()

        if columns.index(self.LABELS) == -1:
            raise Exception('不能缺少标签列:%s' % self.LABELS)
        else:
            columns.remove(self.LABELS)

        self.model_tree = self.__create_tree(data_frame, columns)

    def __create_tree(self, data_frame, columns, level=0):
        """
        构造决策树
        :param data_frame: 训练数据
        :param columns: 剩余列
        :param level 层次
        :return:
        """

        # 初始化当前结点
        current_node = DTNode()

        # 按标签分类统计样本数量
        counter = pd.value_counts(data_frame[self.LABELS])

        # 若数据集只包含一个类别样本 or 到达最大层次数 or 剩余列数为0
        if counter.count() == 1 or level == self.max_level or len(columns)==0:
            current_node.set_target(counter.index[0])
            return current_node


        # 遍历剩余列
        select_column = None # 被选择列
        max_score = None    #当前最大分数
        select_split = None #按被选择列划分数据集
        for column in columns:
            split_data, score = self.estimate(data_frame, column)
            if max_score is None or score > max_score:
                select_column = column
                select_split = split_data
                max_score = score

        current_node.set_column(select_column)
        columns.remove(select_column)
        level = level + 1

        # 递归创建子结点
        for (value, data) in select_split.items():
            child = self.__create_tree(data, columns.copy(),level)
            current_node.add_child(value, child)

        return current_node

    def classify(self, data):
        """
        遍历处理每行
        :param data:
        :return:
        """
        labels = []
        #遍历判别所有分类
        for index,row in data.iterrows():
            labels.append(self.classify_for_row(row))
        return labels

    def classify_for_row(self, row):
        """
        对一条数据进行分类
        :param row: 行数据
        :return:
        """
        current_node = self.model_tree
        # 遍历非叶子结点
        while not hasattr(current_node, 'target'):
            column = current_node.get_column()
            value = row[column]
            current_node = current_node.get_child(value)
        return current_node.get_target()
