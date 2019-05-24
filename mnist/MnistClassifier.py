# -*- coding: utf-8 -*-
"""
@Time: 2019/5/24 13:30
@Author: liulei
@File: MnistClassifier
@desc: mnist - MnistClassifier
"""

import data_loader
import tensorflow as tf

if __name__ == '__main__':
    # 初始化数据集
    mnist = data_loader.read_data_sets('MNIST_data/', one_hot=True)

    # 定义占位符x,y
    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float', [None, 10])

    # 定义模型参数w,b
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # 定义模型输出计算公式,结果大小为None*10
    y_ = tf.nn.softmax(tf.matmul(x, w) + b)

    # 定义损失函数公式
    cross_entropy = -tf.reduce_sum(y*tf.log(y_))

    # 定义最小化模型训练方法
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # 初始化参数
    init = tf.global_variables_initializer()

    # 启用会话
    sess = tf.Session()
    sess.run(init)

    # 循环训练1000次
    for i in range(1000):
        # 读取100张数字图片
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

    # 定义测试集上模型分类正确率的计算公式
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

