# !/usr/bin/python3
# -*-coding:utf-8-*-
# Author: Daphnis
# Github: https://github.com/Daphnis-z
# CreatDate: 2022/1/13 21:10
# Description: 自动摘要
import os
from heapq import nlargest
from itertools import count, product

from nlp.utils import nlp_util


class AutoAbstract(object):
    def __init__(self, use_stopword=True,
                 stop_words_file='etc/stopwords.txt',
                 dict_path=None,
                 max_iter=100,
                 tol=0.0001):
        if dict_path:
            raise RuntimeError("True")
        self.__use_stopword = use_stopword
        self.__dict_path = dict_path
        self.__max_iter = max_iter
        self.__tol = tol

        self.__stop_words = set()
        if stop_words_file:
            self.__stop_words_file = stop_words_file
        if use_stopword:
            for word in open(self.__stop_words_file, 'r', encoding='utf-8'):
                self.__stop_words.add(word.strip())

    def generate_abstract(self, text_content, sentence_num):
        """
        提取文本的主题思想（摘要）
        :param text_content: 文本内容
        :param sentence_num: 取概率最高的前几句话作为摘要返回
        :return: 摘要
        """
        text_content = nlp_util.as_text(text_content)
        tokens = nlp_util.cut_sentences(text_content)
        sentences, sents = nlp_util.cut_filter_words(tokens, self.__stop_words, self.__use_stopword)

        graph = self.create_graph(sents)
        scores = nlp_util.weight_map_rank(graph, self.__max_iter, self.__tol)
        sent_selected = nlargest(sentence_num, zip(scores, count()))
        if sentence_num > len(sent_selected):
            sentence_num = len(sent_selected)
        sent_index = []
        for i in range(sentence_num):
            sent_index.append(sent_selected[i][1])

        result_sent = [sentences[i] for i in sent_index]
        return os.linesep.join(result_sent).replace(' ', '')

    @staticmethod
    def create_graph(word_sent):
        num = len(word_sent)
        board = [[0.0 for _ in range(num)] for _ in range(num)]

        for i, j in product(range(num), repeat=2):
            if i != j:
                board[i][j] = nlp_util.sentences_similarity(word_sent[i], word_sent[j])
        return board
