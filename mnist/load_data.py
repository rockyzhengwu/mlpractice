#!/usr/bin/env python
#-*-coding:utf-8-*-

__author__ = "wu.zheng"

import gzip
import cPickle
import numpy as np

PATH = "../data/mnist.pkl.gz"

def read_data():
    with gzip.open(PATH, 'rb') as f:
        training_data, validation_data, test_data = cPickle.load(f)
        return training_data, validation_data, test_data


def load_data():
    tr_d, va_d, te_d = read_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
