#!/usr/bin/env python
#-*-coding=utf-8-*-

import numpy as np

class Logistic(object):
    def __init__(self, max_steps):
        self.max_steps = max_steps


    def train(self, x, y, theta=0.01):
        n, m = x.shape
        self.w = np.random(m, 1)
        self.b = np.random.random()
        for i in range(self.max_steps):
            h = np.dot(x, self.w) + self.b
            err = self.y - h
            self.w = self.w - theta






