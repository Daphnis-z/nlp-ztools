# !/usr/bin/python3
# -*-coding:utf-8-*-
# Author: Daphnis
# Github: https://github.com/Daphnis-z
# CreatDate: 2022/1/13 21:49
# Description:

import nltk
from datasketch import MinHash

ngrams_num = 3


def calc_similarity(text1, text2):
    """
    计算两段文本的相似度
    :param text1:
    :param text2:
    :return:
    """
    m1, m2 = MinHash(), MinHash()
    for d in nltk.ngrams(text1, ngrams_num):
        m1.update("".join(d).encode('utf8'))
    for d in nltk.ngrams(text2, ngrams_num):
        m2.update("".join(d).encode('utf8'))

    s1 = set(text1)
    s2 = set(text2)
    sty = float(len(s1.intersection(s2))) / float(len(s1.union(s2)))

    return round(sty, 2)
