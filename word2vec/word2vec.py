#!/usr/bin/env python
# -*-coding=utf-8-*-

class VocabWord():
    def __init__(self, word):
        self.word = word
        self.cn = len(word)
        self.code = None
        self.code_len = 0

    def __cmp__(self, other):
        if self.cn >= other.cn:
            return True
        return False

VOCAB = []
VOCAB_HASH = {}
PATH = "/home/rocky/dl/word2vec/trunk/text8"

# PATH = '../data/word2vec_test.txt'

def learn_vocab_from_train_file(file_path):
    fr = open(file_path, "rb")
    b = fr.read(1)
    word = ""
    while b:
        if b == " " or b == "\n":
            index = VOCAB_HASH.get(word, -1)
            if index == -1:
                if word:
                    vocab_word = VocabWord(word)
                    vocab_word.cn = 1
                    VOCAB.append(vocab_word)
                    VOCAB_HASH[word] = len(VOCAB) - 1
                    word = ""
            else:
                VOCAB[index].cn += 1
                word = ""
        else:
            word += b
        b = fr.read(1)


def sorted_vocab():
    sorted(VOCAB)
    for i in range(len(VOCAB)):
        vocab_word = VOCAB[i]
        VOCAB_HASH[vocab_word.word] = i


def create_binary_tree():
    voca_size = len(VOCAB)
    count = [0] * (2 * voca_size + 1)
    binary = [0] * (2 * voca_size + 1)
    parent_node = [0] * (2 * voca_size + 1)
    ## 初始化count
    for i in range(voca_size):
        count[i] = VOCAB[i].cn

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
        VOCAB[i].codelen = a
        VOCAB[i].code = "".join(code)
    print "finish create tree"

WINDOW = 5

def train_one_thread(num):
    pass


def train_model():
    learn_vocab_from_train_file(PATH)
    print "finish_learn"
    sorted_vocab()
    print "sorted"
    create_binary_tree()
    print "create_tree"

if __name__ == '__main__':
    train_model()
