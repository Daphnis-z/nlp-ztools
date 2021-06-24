# !/usr/bin/python3
# -*-coding:utf-8-*-
# Author: Daphnis
# Github: https://github.com/Daphnis-z
# CreatDate: 2021/6/22 22:24
# Description: NLP算法样例代码
from nlp.keyword.keyword_extration import KeywordExtraction
from nlp.utils import file_util


def keyword_demo():
    kw_extract = KeywordExtraction(stopword_file='etc/stopwords.txt', keyword_weight=0.25)

    content = file_util.read_whole_file('data/test001.txt')
    keyword_list = kw_extract.extract_keyword(content)

    print('extract keywords: {}'.format(keyword_list))
    # print('extract keywords:')
    # [print(word_weight[0] + ': ' + word_weight[1]) for word_weight in keyword_list]


if __name__ == '__main__':
    keyword_demo()
