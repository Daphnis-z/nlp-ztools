import jieba
from jieba import analyse


class KeywordExtraction:

    def __init__(self, stopword_file, keyword_weight=0.2):
        """
        初始化
        :param stopword_file: 停用词文件路径
        :param keyword_weight: 关键词权重，权重大于此值的关键词才会被返回
        """
        # 添加自定义词典
        jieba.load_userdict('etc/user_words.dict')

        # 设置停用词，内置了一些，还可以根据自己的需要进行添加
        analyse.set_stop_words(stopword_file)

        # 指定哪些词性的词可以成为关键词
        # eng: 英文单词；x：自定义词；nr：人名
        self.allow_pos = ('ns', 'n', 'vn', 'v', 'eng', 'x', 'nr')

        self.keyword_weight = keyword_weight

    def extract_keyword(self, content, keyword_num=20):
        """
        提取目标文本中指定数量的关键词
        :param content: 目标文本
        :param keyword_num: 返回的关键词数量
        :return: 关键词列表
        """
        result = analyse.textrank(content, topK=keyword_num, withWeight=True, allowPOS=self.allow_pos)

        keyword_list = []
        for item in result:
            # 为了增强关键词的准确性，只保留权重大于一定值的关键词
            if item[1] > self.keyword_weight:
                keyword_list.append(item[0])

        return keyword_list
