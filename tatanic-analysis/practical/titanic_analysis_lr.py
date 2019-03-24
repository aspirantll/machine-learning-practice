# -*- coding: utf-8 -*-
"""
@Time: 2018/3/26 0026
@Author: ll
@File: titanic_analysis
@desc: 泰坦尼克号幸存者分析
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV
import sklearn.model_selection as ms
import sklearn.metrics as mt
import sklearn.decomposition as ds
import numpy as np
import time


# 预处理
def preprocess(data_set):
    drop_col_names = ['PassengerId', 'Name', 'Ticket', 'Cabin']

    prop_col = data_set.drop(drop_col_names, axis=1)

    # 性别替换为数值
    prop_col['Sex'] = prop_col['Sex'].map(str.lower).map({'female': 0, 'male': 1})

    # 补全登船地点缺失值
    prop_col['Embarked'] = prop_col['Embarked'].map({'S': 0, 'C': 1, 'Q':2})

    # 补全登船地点缺失值
    prop_col['Embarked'] = prop_col['Embarked'].fillna(prop_col['Embarked'].mode().values[0])

    # 兄弟姐妹数量标准化
    sib_sp = preprocessing.scale(prop_col['SibSp'])

    # 父母子女数量标准化
    parch = preprocessing.scale(prop_col['Parch'])

    # 费用标准化
    prop_col['Fare'] = prop_col['Fare'].fillna(0)
    fare = preprocessing.scale(prop_col['Fare'])

    # 补全年龄缺失值
    age = preprocessing.scale(prop_col['Age'].fillna(prop_col['Age'].median()))


    prop_col = prop_col.drop(['SibSp', 'Parch', 'Fare', 'Age'], axis=1)

    # one-hot
    enc = preprocessing.OneHotEncoder()
    enc_prop_col = enc.fit_transform(prop_col).toarray()

    return np.column_stack((enc_prop_col, sib_sp, parch, fare, age))

if __name__ == '__main__':
    file_path = r'C:\Users\Administrator\Downloads\train.csv'
    test_path = r'C:\Users\Administrator\Downloads\test.csv'
    label_col = "Survived"
    data_set = pd.read_csv(file_path)

    # 区分标签列和属性列
    x, y = data_set.drop('Survived', axis=1), data_set['Survived']
    # 属性值预处理
    enc_x = preprocess(x)
    print(enc_x)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = ms.train_test_split(enc_x, y, random_state=1)

    # 利用训练集训练模型
    logistic_model = LogisticRegressionCV()
    logistic_model.fit(enc_x, y)

    # 测试集上预测效果
    y_test_hat = logistic_model.predict(x_test)
    # 准确率
    ac_score = mt.accuracy_score(y_test, y_test_hat)
    # f1 score
    f1_score = mt.f1_score(y_test, y_test_hat)
    # 召回率
    re_score = mt.recall_score(y_test, y_test_hat)

    print('f1分数:%s, 准确率:%s， 召回率:%s' % (f1_score, ac_score, re_score))

    test_set = pd.read_csv(test_path)
    enc_x_test = preprocess(test_set)

    test_label_hat = logistic_model.predict(enc_x_test)

    predict_result = pd.DataFrame(np.column_stack((test_set['PassengerId'].values, test_label_hat)), columns=['PassengerId','Survived'])

    predict_result.to_csv(r'C:\Users\Administrator\Downloads\result' + time.strftime('%Y%m%d%H%M%S') + '.csv', index=False)







