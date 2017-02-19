#!/usr/bin/env python
# -*-coding=utf-8-*-

"""
multilayer feedforward nueral network

"""

import numpy as np
from load_data import load_data


def sigmoid(z):
    return 1.0 / (1 + np.exp(-1.0 * z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class MFNN(object):
    def __init__(self, sizes):
        """

        :param sizes:  the size fo layer example [784, 200, 10] 784输入层，一个隐藏层，200个神经元，输出层10个
        """
        self.w = [np.random.randn(r, l) for r, l in zip(sizes[1:], sizes[:-1])]
        self.b = [np.random.randn(r, 1) for r in sizes[1:]]
        self.num_layers = len(sizes)

    def backprop(self, x, y):
        Z = []
        A = []
        # forward
        n_dw = [np.zeros(tw.shape) for tw in self.w]
        n_db = [np.zeros(tw.shape) for tw in self.b]
        input = x
        A.append(input)
        for w, b in zip(self.w, self.b):
            z = np.dot(w, input) + b
            Z.append(z)
            input = sigmoid(z)
            A.append(input)
        # backprob

        # 计算最后一层对在d的导数, 放到delta_list
        delta_list = [0 * i for i in range(self.num_layers - 1)]
        delta = (A[-1] - y ) * sigmoid_derivative(z)
        delta_list[-1] = delta

        # 计算最后一层对w,b 的偏导数
        n_db[-1] = delta
        n_dw[-1] = delta.dot(A[-2].transpose())
        # 误差反向传播
        for layer in range(2, self.num_layers):
            z = Z[-layer]
            # 后一层的w
            w = self.w[-layer + 1]
            delta_last = delta_list[-layer + 1]
            # delta 的传递
            delta = w.transpose().dot(delta_last) * sigmoid_derivative(z)
            delta_list.append(delta)
            # 计算对b和w 的偏导数
            db = delta
            dw = delta.dot(A[-layer - 1].transpose())
            n_db[-layer] = db
            n_dw[-layer] = dw
        return n_db, n_dw

    def train_gd(self, X, Y, test_x, test_y, epochs=100, rate=0.01):

        m = len(X)
        for epoch in range(epochs):
            dw = [np.zeros(t.shape) for t in self.w]
            db = [np.zeros(t.shape) for t in self.b]

            for x, y in zip(X, Y):
                n_db, n_dw = self.backprop(x, y)
                dw = [a + b for a, b in zip(dw, n_dw)]
                db = [a + b for a, b in zip(db, n_db)]
            self.w = [w - (rate / m) * d for w, d in zip(self.w, dw)]
            self.b = [b - (rate / m) * d for b, d in zip(self.b, db)]
            counter = self.test(test_x, test_y)
            print "epoch %d %d/%d" % (epoch, counter, len(test_y))

    def update_batch(self, X, Y, rate):
        batch_dw = [np.zeros(t.shape) for t in self.w]
        batch_db = [np.zeros(t.shape) for t in self.b]
        for x, y in zip(X, Y):
            n_db, n_dw = self.backprop(x, y)
            batch_dw = [a + b for a, b in zip(batch_dw, n_dw)]
            batch_db = [a + b for a, b in zip(batch_db, n_db)]

        m = len(X)
        self.w = [w - (rate / m) * d for w, d in zip(self.w, batch_dw)]
        self.b = [b - (rate / m) * d for b, d in zip(self.b, batch_db)]

    def train_batch(self, X, Y, test_x, test_y, epochs=100, mini_batch=10, rate=0.01):
        m = len(X)
        batch_x = [X[k:k + mini_batch] for k in xrange(0, m, mini_batch)]
        batch_y = [Y[k:k + mini_batch] for k in xrange(0, m, mini_batch)]
        for epoch in xrange(epochs):
            for b_x, b_y in zip(batch_x, batch_y):
                self.update_batch(b_x, b_y, rate)
            counter = self.test(test_x, test_y)
            print "epoch: %d/%d" % (counter, len(test_x))

    def test(self, X, Y):
        counter = 0
        for x, y in zip(X, Y):
            pre = self.predict(x)
            if pre == y:
                counter += 1
        return counter

        self.predict(X)

    def predict(self, a):
        for w, b in zip(self.w, self.b):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            r = np.argmax(a)
        return r


if __name__ == '__main__':
    train_data, validation_data, test_data = load_data()
    sizes = [784, 30, 10]
    nn = MFNN(sizes)
    train_x = [t[0] for t in train_data]
    train_y = [t[1] for t in train_data]
    test_x = [t[0] for t in test_data]
    test_y = [t[1] for t in test_data]

    nn.train_gd(train_x, train_y, test_x, test_y, rate=3.0)

    # nn.train_batch(train_x, train_y, test_x, test_y, rate=3.0)
