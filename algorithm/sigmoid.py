#!/usr/bin/env python
#-*-coding=utf-8-*-


import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))
