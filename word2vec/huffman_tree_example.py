#!/usr/bin/env python
# -*-coding=utf-8-*-

"""
通过一个例子说明huffman树的构造过程
"""


class VocabWord():
    def __init__(self, word, cn=0):
        self.word = word
        self.cn = cn
        self.code = None
        self.codelen = 0


vocab = [
    VocabWord("我", 15),
    VocabWord("喜欢", 8),
    VocabWord("观看", 6),
    VocabWord("巴西", 5),
    VocabWord("足球", 3),
    VocabWord("世界杯", 1)
]


def create_binary_tree():
    voca_size = len(vocab)
    count = [0] * (2 * voca_size + 1)
    binary = [0] * (2 * voca_size + 1)
    parent_node = [0] * (2 * voca_size + 1)
    ## 初始化count
    for i in range(voca_size):
        count[i] = vocab[i].cn

    for i in range(voca_size, 2 * voca_size + 1):
        count[i] = 1e15
    pos1 = voca_size - 1
    pos2 = voca_size
    for i in range(voca_size):
        if (pos1 >= 0):
            if count[pos1] < count[pos2]:
                min1i = pos1
                pos1 -= 1
            else:
                min1i = pos2
                pos2 += 1
        else:
            min1i = pos2
            pos2 += 1

        if (pos1 >= 0):
            if count[pos1] < count[pos2]:
                min2i = pos1
                pos1 -= 1
            else:
                min2i = pos2
                pos2 += 1
        else:
            min2i = pos2
            pos2 += 1

        count[voca_size + i] = count[min1i] + count[min2i]
        parent_node[min1i] = voca_size + i
        parent_node[min2i] = voca_size + i
        binary[min2i] = 1;

    for i in range(voca_size):
        j = i
        a = 0
        code = []
        while True:
            code.append(str(binary[j]))
            j = parent_node[j]
            a += 1
            if (j == voca_size * 2 - 2):
                break
        vocab[i].codelen = a
        vocab[i].code = "".join(code)

    print "finish create tree"


if __name__ == '__main__':
    create_binary_tree()
    print "finish all"
