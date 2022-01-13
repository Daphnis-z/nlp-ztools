import re
from math import log

from nlp.abstract.perceptron import Perceptron

re_eng = re.compile('[a-zA-Z0-9]', re.U)
re_han = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)
re_skip = re.compile("(\r\n|\s)", re.U)


class Segment:
    def __init__(self):
        self.vocab = {}
        self.max_word_len = 0
        self.max_freq = 0
        self.total_freq = 0
        self.initialized = False
        self.model = None

    def init(self, vocab_path='data/abstract/dict/jiagu.dict', user_vocab='data/abstract/dict/user.dict',
             model_path='data/abstract/model/cws.model'):
        self.load_vocab(vocab_path)
        self.load_vocab(user_vocab)
        self.model = Perceptron(model_path)
        self.initialized = True

    def load_vocab(self, vocab_path):
        fin = open(vocab_path, 'r', encoding='utf8')
        for index, line in enumerate(fin):
            line = line.strip()
            if line == '':
                continue
            word_freq_tag = line.split('\t')
            if len(word_freq_tag) == 1:
                word = word_freq_tag[0]
                self.add_vocab(word)
            elif len(word_freq_tag) == 2:
                word = word_freq_tag[0]
                freq = int(word_freq_tag[1])
                self.add_vocab(word, freq)
        fin.close()

    def add_vocab(self, word=None, freq=None, tag=None):
        if freq == None:
            freq = self.max_freq

        if word not in self.vocab:
            self.vocab[word] = 0

        self.vocab[word] += freq
        self.total_freq += freq

        if freq > self.max_freq:
            self.max_freq = freq

        if len(word) > self.max_word_len:
            self.max_word_len = len(word)

    def del_vocab(self, word=None, freq=None, tag=None):
        if word not in self.vocab:
            return None

        vocab_freq = self.vocab[word]
        if freq == None or vocab_freq <= freq:
            del self.vocab[word]
            self.total_freq -= vocab_freq
        else:
            self.vocab[word] -= freq

    # self.max_freq and self.max_word_len ?

    def load_userdict(self, userdict):
        if self.initialized == False:
            self.init()

        if isinstance(userdict, str):
            self.load_vocab(userdict)

        for item in userdict:
            if isinstance(item, list):
                if len(item) == 1:
                    word = item[0]
                    self.add_vocab(word)
                elif len(item) == 2:
                    word = item[0]
                    freq = item[1]
                    self.add_vocab(word, freq)
            elif isinstance(item, str):
                self.add_vocab(word=item)

    def del_userdict(self, userdict):
        if self.initialized == False:
            self.init()

        for item in userdict:
            if isinstance(item, list):
                if len(item) == 1:
                    word = item[0]
                    self.del_vocab(word)
                elif len(item) == 2:
                    word = item[0]
                    freq = item[1]
                    self.del_vocab(word, freq)
            elif isinstance(item, str):
                self.del_vocab(word=item)

    def calc_route(self, sentence, DAG, route):
        vocab = self.vocab
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = log(self.total_freq)
        for idx in range(N - 1, -1, -1):
            route[idx] = max(
                (log(vocab.get(sentence[idx:x + 1]) or 1) - logtotal + route[x + 1][0], x) for x in DAG[idx])

    def create_DAG(self, sentence):
        vocab = self.vocab
        max_word_len = self.max_word_len
        DAG = {}
        N = len(sentence)
        for idx in range(N):
            cand_idx = [idx]
            for i in range(idx + 1, idx + min(max_word_len, N - idx), 1):
                cand = sentence[idx: i + 1]
                if cand in vocab:
                    cand_idx.append(i)
            DAG[idx] = cand_idx
        return DAG

    def cut_search(self, sentence):
        DAG = self.create_DAG(sentence)
        old_j = -1
        for k, L in DAG.items():
            if len(L) == 1 and k > old_j:
                yield sentence[k:L[0] + 1]
                old_j = L[0]
            else:
                for j in L:
                    if j > k:
                        yield sentence[k:j + 1]
                        old_j = j

    def cut_vocab(self, sentence):
        DAG = self.create_DAG(sentence)
        route = {}
        self.calc_route(sentence, DAG, route)

        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if buf:
                yield buf
                buf = ''
            yield l_word
            x = y
        if buf:
            yield buf
            buf = ''

    def cut_words(self, sentence):
        DAG = self.create_DAG(sentence)
        route = {}
        self.calc_route(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    yield buf
                    buf = ''
                yield l_word
                x = y
        if buf:
            yield buf
            buf = ''

    def model_cut(self, sentence):
        if sentence == '':
            return ['']

        sentence = list(sentence)
        labels = self.model.predict(sentence)
        return self.__lab2word(sentence, labels)

    def __lab2word(self, sentence, labels):
        sen_len = len(sentence)
        tmp_word = ""
        words = []
        for i in range(sen_len):
            label = labels[i]
            w = sentence[i]
            if label == "B":
                tmp_word += w
            elif label == "M":
                tmp_word += w
            elif label == "E":
                tmp_word += w
                words.append(tmp_word)
                tmp_word = ""
            else:
                if tmp_word != '':
                    words.append(tmp_word)
                    tmp_word = ""
                words.append(w)
        if tmp_word:
            words.append(tmp_word)
        return words

    def seg_default(self, sentence):
        blocks = re_han.split(sentence)
        cut_block = self.cut_words
        cut_all = False
        for block in blocks:
            if not block:
                continue
            if re_han.match(block):
                for word in cut_block(block):
                    yield word
            else:
                tmp = re_skip.split(block)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    elif not cut_all:
                        for xx in x:
                            yield xx
                    else:
                        yield x

    def seg_new_word(self, sentence):
        blocks = re_han.split(sentence)
        cut_block = self.cut_words
        cut_all = False
        for block in blocks:
            if not block:
                continue
            if re_han.match(block):
                words1 = list(cut_block(block))
                # print(words1)

                words2 = self.model_cut(block)
                # print(words2)

                new_word = []  # 有冲突的不加，长度大于4的不加，加完记得删除
                length = len(words1)
                for n in range(3):
                    can_limit = length - n + 1
                    for i in range(0, can_limit):
                        ngram = ''.join(words1[i:i + n])
                        word_len = len(ngram)
                        if word_len > 4 or word_len == 1:
                            continue
                        if ngram in words2 and ngram not in words1:
                            # print(ngram)
                            new_word.append([ngram, 1])

                # new_word = []
                # for word in words2:
                # 	if word not in words1 and len(word)>1 and len(word) < 4 :#and not re_eng.match(word):
                #		new_word.append([word, 1])

                self.load_userdict(new_word)

                # print('------------------')

                for word in cut_block(block):
                    yield word

                # 删除字典
                self.del_userdict(new_word)


            else:
                tmp = re_skip.split(block)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    elif not cut_all:
                        for xx in x:
                            yield xx
                    else:
                        yield x

    def seg(self, sentence, mode="default"):
        if self.initialized == False:
            self.init()

        if mode == 'probe':
            return list(self.seg_new_word(sentence))
        else:
            return list(self.seg_default(sentence))
