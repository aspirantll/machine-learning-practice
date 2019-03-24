# -*- coding: utf-8 -*-
"""
@Time: 2018/7/21 0021
@Author: ll
@File: test
@desc:
"""

from DecisionTree import model
import pandas as pd
import numpy as np


def divide_ts(total_data):
    data_matrix = total_data[:, 1:]
    data_frame = pd.DataFrame(data=data_matrix, columns=['age', 'spectacle', 'astigmatic', 'tear', 'label'], index = total_data[:, 0])
    count_series = data_frame['label'].groupby(data_frame['label']).count()
    example_count = pd.Series(data=(count_series.values * 3)//4, index=count_series.index)

    example_data = pd.DataFrame(columns=data_frame.columns)
    test_data = pd.DataFrame(columns=data_frame.columns)
    for index, row in data_frame.iterrows():
        target = row['label']
        remain_count = example_count.get_value(target)
        if remain_count > 0:
            example_data = example_data.append(row)
            remain_count = remain_count - 1
            example_count.set_value(target, remain_count)
        else:
            test_data = test_data.append(row)

    return example_data, test_data


if __name__ == '__main__':
    data = np.loadtxt('lenses')
    example, test = divide_ts(data)
    decision_tree = model.DTModel()
    decision_tree.fit(example)
    result = decision_tree.classify(test)
    compare = test['label'] == result
    print(compare)
    print(compare.sum()/len(compare))