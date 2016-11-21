#!/usr/bin/env python
# -*-coding=utf-8-*-

import numpy as np

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


if __name__ == '__main__':
    test_knn()
