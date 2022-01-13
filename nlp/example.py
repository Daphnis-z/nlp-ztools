# !/usr/bin/python3
# -*-coding:utf-8-*-
# Author: Daphnis
# Github: https://github.com/Daphnis-z
# CreatDate: 2021/6/22 22:24
# Description: NLP算法样例代码
from nlp.abstract.auto_abstract import AutoAbstract
from nlp.entity import named_entity
from nlp.keyword.keyword_extration import KeywordExtraction
from nlp.similarity import text_similarity
from nlp.utils import file_util


def keyword_demo():
    """ 关键词提取 """

    kw_extract = KeywordExtraction(stopword_file='etc/stopwords.txt', keyword_weight=0.25)

    content = file_util.read_whole_file('data/test001.txt')
    keyword_list = kw_extract.extract_keyword(content)

    print('extract keywords: {}'.format(keyword_list))
    print('--' * 55)


def named_entity_demo():
    """ 命名实体识别 """
    # 目前只能提取常见的三种命名实体：人名、地名和组织机构名
    content = file_util.read_whole_file('data/test001.txt')
    entities = named_entity.extract_entity(content)

    print('extract named entities: {}'.format(entities))
    print('--' * 55)


def abstract_demo():
    """ 自动摘要 """

    content = file_util.read_whole_file('data/test005.txt')
    abstract = AutoAbstract().generate_abstract(content, 3)

    print('generate abstract: ')
    print('  ' + abstract)
    print('--' * 55)


def text_similarity_demo():
    """ 文本相似度 """
    file1 = 'data/test001.txt'
    file2 = 'data/test002.txt'
    file3 = 'data/test005.txt'
    text1 = file_util.read_whole_file(file1)
    text2 = file_util.read_whole_file(file2)
    text3 = file_util.read_whole_file(file3)

    sty1 = text_similarity.calc_similarity(text1, text2)
    sty2 = text_similarity.calc_similarity(text1, text3)

    print("{} and {} similarity: {}".format(file1, file2, sty1))
    print("{} and {} similarity: {}".format(file1, file3, sty2))
    print('--' * 55)


if __name__ == '__main__':
    keyword_demo()
    named_entity_demo()
    abstract_demo()
    text_similarity_demo()
