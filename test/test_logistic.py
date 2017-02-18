#!/usr/bin/env python
#-*-coding=utf-8-*-


def load_data():
    data_path = "../data/logistic_test_data.txt"
    data_set = []
    labels = []
    for lin in open(data_path):
        lin = lin.strip("\n")
        lin = lin.split("\t")
        data_set.append(lin[:1])
        labels.append(lin[-1])
    return data_set, labels

if __name__ == '__main__':
    data_set, labels = load_data()
    print (data_set)

