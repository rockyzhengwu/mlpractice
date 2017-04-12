#!/usr/bin/env python
# -*-coding=utf-8-*-

"""
计算卷积层
"""


import numpy  as np

input = [
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
]

filter = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
]

input_array = np.array(input)
filter_array = np.array(filter)

filter_w = filter_array.shape[0]
filter_h = filter_array.shape[1]
stride = 2

out_w = (input_array.shape[0] - filter_w + 0) / stride + 1
out_h = (input_array.shape[1] - filter_h + 0) / stride + 1
out_put_array = np.zeros((out_w, out_h))

for i in range(0, out_w):
    for j in range(0, out_h):
        input_batch = input_array[i*stride :i*stride + filter_w:, j*stride:j*stride + filter_h:]
        out_put_array[i][j] = np.sum(input_batch * filter_array)


print out_put_array