# !/usr/bin/python3
# -*-coding:utf-8-*-
# Author: Daphnis
# Github: https://github.com/Daphnis-z
# CreatDate: 2022/1/13 21:12
# Description:
import codecs
import math

from jieba import posseg

from nlp.abstract.nroute import Segment


def different(scores, old_scores, tol=0.0001):
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= tol:  # 原始是0.0001
            flag = True
            break
    return flag


def sentences_similarity(s1, s2):
    """计算两个句子的相似度

	:param s1: list
	:param s2: list
	:return: float
	"""
    counter = 0
    for sent in s1:
        if sent in s2:
            counter += 1
    if counter == 0:
        return 0
    return counter / (math.log(len(s1) + len(s2)))


def cut_words(text, stop_flag=['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']):
    """ 将文本切成词语 """
    stop_words = codecs.open('etc/stopwords.txt', 'r', encoding='utf8').readlines()
    stop_words = [w.strip() for w in stop_words]

    result = []
    words = posseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stop_words:
            result.append(word)
    return result


def cut_sentences(text):
    """
    将文本切成句子
    :param text: 待切分文本
    :return: 句子列表
    """
    text = text.translate(str.maketrans('!?！？\r\n', '。。。。  '))
    return [sen for sen in text.split('。') if len(sen.strip()) > 1]


def as_text(v):
    """生成unicode字符串"""
    if v is None:
        return None
    elif isinstance(v, bytes):
        return v.decode('utf-8', errors='ignore')
    elif isinstance(v, str):
        return v
    else:
        raise ValueError('Unknown type %r' % type(v))


def cut_filter_words(cutted_sentences, stopwords, use_stopwords=False):
    seg = Segment()

    sentences = []
    sents = []
    for sent in cutted_sentences:
        sentences.append(sent)
        if use_stopwords:
            sents.append([word for word in seg.seg(sent) if word and word not in stopwords])  # 把句子分成词语
        else:
            sents.append([word for word in seg.seg(sent) if word])
    return sentences, sents


def weight_map_rank(weight_graph, max_iter, tol):
    # 初始分数设置为0.5
    # 初始化每个句子的分子和老分数
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]
    denominator = get_degree(weight_graph)

    # 开始迭代
    count = 0
    while different(scores, old_scores, tol):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        # 计算每个句子的分数
        for i in range(len(weight_graph)):
            scores[i] = get_score(weight_graph, denominator, i)
        count += 1
        if count > max_iter:
            break
    return scores


def get_degree(weight_graph):
    length = len(weight_graph)
    denominator = [0.0 for _ in range(len(weight_graph))]
    for j in range(length):
        for k in range(length):
            denominator[j] += weight_graph[j][k]
        if denominator[j] == 0:
            denominator[j] = 1.0
    return denominator


def get_score(weight_graph, denominator, i):
    """

	:param weight_graph:
	:param denominator:
	:param i: int
		第i个句子
	:return: float
	"""
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0

    for j in range(length):
        # [j,i]是指句子j指向句子i
        fraction = weight_graph[j][i] * 1.0
        # 除以j的出度
        added_score += fraction / denominator[j]
    weighted_score = (1 - d) + d * added_score
    return weighted_score
