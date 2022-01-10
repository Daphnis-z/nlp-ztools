# !/usr/bin/python3
# -*-coding:utf-8-*-
# Author: Daphnis
# Github: https://github.com/Daphnis-z
# CreatDate: 2021/6/22 22:24
# Description: NLP算法样例代码
from nlp.entity import named_entity
from nlp.keyword.keyword_extration import KeywordExtraction
from nlp.utils import file_util


def keyword_demo():
    """ 关键词提取 """

    kw_extract = KeywordExtraction(stopword_file='etc/stopwords.txt', keyword_weight=0.25)

    content = file_util.read_whole_file('data/test001.txt')
    keyword_list = kw_extract.extract_keyword(content)

    print('extract keywords: {}'.format(keyword_list))


def named_entity_demo():
    """ 命名实体识别 """
    # 目前只能提取常见的三种命名实体：人名、地名和组织机构名
    content = file_util.read_whole_file('data/test001.txt')
    entities = named_entity.extract_entity(content)

    print('extract named entities: {}'.format(entities))


def abstract_demo():
    """ 自动摘要 """

    pass


def text_similarity_demo():
    """ 文本相似度 """

    pass


if __name__ == '__main__':
    keyword_demo()
    named_entity_demo()
