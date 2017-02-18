#!/usr/bin/env python
# -*-coding=utf-8-*-

import numpy as np
import os

from algorithm.knn import KNN


def load_data():
    group = np.array(
        [
            [1.0, 1.1],
            [1.0, 1.0],
            [0, 0],
            [0, 0.1]
        ]
    )
    labels = ["A", "A", "B", "B"]
    return group, labels


def test_knn():
    model = KNN(3)
    x, y = load_data()
    model.fit(x, y)
    result = model.predict([0, 0])
    print(result)


def hand_write():
    base_path = "/home/rocky/Documents/book/machinelearninginaction/Ch02/digits"
    x = []
    y = []
    train_path = os.path.join(base_path, "trainingDigits")
    for ts in os.listdir(train_path):
        y.append(ts.split("_")[0])
        tmp_list = []
        for lin in open(os.path.join(train_path, ts)):
            lin = lin.strip("\n")
            if not lin:
                continue
            tmp_list.extend(float(num) for num in list(lin))
        x.append(tmp_list)
    x = np.array(x)

    model = KNN(3)
    model.fit(x, y)
    test_path = os.path.join(base_path, "testDigits")
    acc_count = 0
    counter = 0
    for ts in os.listdir(test_path):
        counter += 1
        test_x = []
        test_y = ts.split("_")[0]
        for lin in open(os.path.join(train_path, ts)):
            lin = lin.strip("\n")
            if not lin:
                continue
            test_x.extend(float(num) for num in list(lin))
        test_x = np.array(test_x)
        test_x.reshape(1024, 1)

        predict_y = model.predict(np.array(test_x))
        print("redict:%s, acture:%s" % (predict_y, test_y))
        if predict_y == test_y:
            acc_count += 1
        print("acc:%.4f" % (acc_count / counter))


if __name__ == '__main__':
    # test_knn()
    hand_write()
