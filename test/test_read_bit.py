#!/usr/bin/env python
#-*-coding=utf-8-*-

path = "/home/rocky/dl/word2vec/trunk/text8"
# path = "test"
f = open(path, 'r')
word = ""
a = f.read(1)
import time

while a:
    if a not in [" ", "\t", "\n"] :
        word += a
    else:
        print word
        word = ""
    a = f.read(1)