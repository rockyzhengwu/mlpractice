#!/usr/bin/env python
#-*-coding:utf-8-*-

__author__ = "wu.zheng"


import gzip
import numpy as np
from load_data import load_data

train_data, validation_data, test_data = load_data()
print type(train_data)
print len(train_data)


class SoftmaxClassfier():
    def __init__(self):
        self.w = None
        self.b = None

    def init_param(self, nf, nt):
        self.w = np.random.randn(nt, nf)
        self.b = np.random.randn(nt, 1)

    def soft_max(self, z):
        ex_z = np.exp(z)
        s = np.sum(ex_z)
        return ex_z/s


    def cross_entry(self,y, h):
        m = len(y)
        j = np.log(h)*y  + (1-y)*np.log(1-h)
        return -1/m*np.sum(j)


    def train(self, X, Y, epochs):
        nf = X[0].shape[0]  # 样本特征数
        nt = Y[0].shape[0]  # 分类数
        self.init_param(nf, nt)

        err = 0
        for epoch in xrange(epochs):
            for x,y in zip(X, Y):
                z = self.w.dot(x) + self.b
                h = self.soft_max(z)
                err += self.cross_entry(y, h)
                # 计算梯度，更新w,b



            pass


