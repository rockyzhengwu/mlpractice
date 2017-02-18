#!/usr/bin/env python
# -*-coding:utf-8-*-

__author__ = "wu.zheng"

import numpy as np
from load_data import load_data


class SoftmaxClassfier():
    def __init__(self):
        self.w = None
        self.b = None

    def init_param(self, nf, nt):
        self.w = np.random.randn(nt, nf)
        self.b = np.random.randn(nt, 1)

    def soft_max(self, z):
        ex_z = np.exp(z)
        h = ex_z / np.sum(ex_z)
        return h

    def cross_entry_d(self, y, h):
        j = np.log(h) * y + (1 - y) * np.log(1 - h)
        return np.sum(j)

    def cross_entry(self, y, h):
        return y * h

    def trainGD(self, X, Y, test_x, test_y, epochs):
        """
        随机梯度下降优化
        :param X: [np.array(781,1),np.array(784,1)]
        :param Y: [np.array(10,1), np.array(10,1)]
        :param test_x: same as X
        :param test_y: [3,4,5]
        :param epochs: 迭代次数
        :return:
        """
        nf = X[0].shape[0]  # 样本特征数
        nt = Y[0].shape[0]  # 分类数
        rate = 0.01  # 更新速率
        lamda = 0.01

        self.init_param(nf, nt)

        for epoch in range(epochs):
            loss = 0  # 误差
            d_w = np.zeros(self.w.shape)  # 误差关于w的偏导
            d_b = np.zeros(self.b.shape)  # 误差关于b的偏导

            # todo question: 下面这个循环能用矩阵运算替代？
            for x, y in zip(X, Y):
                # x: 784, 1
                # w: 10,784
                z = np.dot(self.w, x) + self.b
                h = self.soft_max(z)
                loss += np.sum(y * np.log(h))

                dl = np.dot(x, (y - h).transpose())
                d_w += dl.transpose()
                d_b += (y - h)
            # 梯度计算
            d_h = (-1.0 / nf) * d_w + lamda * self.w
            d_b = (-1.0 / nf) * d_b
            loss = (-1.0 / nf) * loss + np.sum(lamda / 2.0 * (self.w * self.w))

            self.w -= rate * d_h
            self.b -= rate * d_b
            print ("epoch %d losss: %f" % (epoch, loss))
            print("epoch %d %d/%d" % (epoch, self.test(test_x, test_y), len(test_y)))

    def trainSGD(self, X, Y, test_x, test_y, epochs):
        """
        :return:
        """
        nf = X[0].shape[0]  # 样本特征数
        nt = Y[0].shape[0]  # 分类数
        rate = 0.01  # 更新速率
        lamda = 0.01
        self.init_param(nf, nt)

        for epoch in range(epochs):
            loss = 0  # 误差

            for x, y in zip(X, Y):
                # x: 784, 1
                # w: 10,784

                z = np.dot(self.w, x) + self.b
                h = self.soft_max(z)
                # 误差
                loss += np.sum(y * np.log(h))  # cross entry

                #  计算梯度
                dl = np.dot(x, (y - h).transpose())
                d_w = -1 * dl.transpose()
                d_b = -1 * (y - h)

                # 更新参数
                self.w -= rate * d_w + lamda * self.w
                self.b -= rate * d_b

            loss = (-1.0 / nf) * loss + np.sum(lamda / 2.0 * (self.w * self.w))
            print ("epoch %d losss: %f" % (epoch, loss))
            print("epoch %d %d/%d" % (epoch, self.test(test_x, test_y), len(test_y)))

    def test(self, test_x, test_y):
        counter = 0
        for x, y in zip(test_x, test_y):
            pre = self.predict(x)
            if pre == y:
                counter += 1
        return counter

    def predict(self, x):
        z = np.dot(self.w, x) + self.b
        pre = self.soft_max(z)
        return np.argmax(pre)


if __name__ == '__main__':

    train_data, validation_data, test_data = load_data()

    train_x = [t[0] for t in train_data]
    train_y = [t[1] for t in train_data]

    validation_x = [t[0] for t in validation_data]
    validation_y = [t[1] for t in validation_data]

    test_x = [t[0] for t in test_data]
    test_y = [t[1] for t in test_data]

    softclass = SoftmaxClassfier()
    softclass.trainGD(train_x, train_y, test_x, test_y, 500)
    softclass.trainSGD(train_x, train_y, test_x, test_y, 200)
