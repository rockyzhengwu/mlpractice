#!/usr/bin/env python
#-*-coding=utf-8-*-

import numpy as np
from collections import defaultdict

class KNN(object):

    def __init__(self, n_neighbors):

        self.n_neighbors = n_neighbors


    def fit(self, x, y):
        self.data_set = x
        self.labels = y
        self.siample_size = self.data_set.shape[0]


    def predict(self, x ):

        t_x = np.tile(x, (self.siample_size, 1))
        diff_mat = t_x - self.data_set

        distances = np.sum(diff_mat**2, axis=1)**0.5
        sort_index = distances.argsort()
        class_count = defaultdict(int)
        for i in range(self.n_neighbors):
            label = self.labels[sort_index[i]]
            class_count[label] += 1
        sort_class_count = sorted(class_count.items(),  reverse=True)
        return sort_class_count[0][0]
