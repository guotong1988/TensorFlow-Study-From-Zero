#encoding:utf-8

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools

class TextLoader2():
    def __init__(self, data_dir, batch_size, seq_length, mini_frq=5):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.mini_frq = mini_frq

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        self.preprocess(input_file, vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences):
        word_counts = collections.Counter()
        if not isinstance(sentences, list):
            sentences = [sentences]
        for sent in sentences:
            word_counts.update(sent)
        vocabulary_inv = ['<START>', '<UNK>', '<END>'] + \
                         [x[0] for x in word_counts.most_common() if x[1] >= self.mini_frq]
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            if lines[0][:1] == codecs.BOM_UTF8:
                lines[0] = lines[0][1:]
            lines = [line.strip().split() for line in lines]

        self.vocab, self.words = self.build_vocab(lines)
        self.vocab_size = len(self.words)
        #print 'word num: ', self.vocab_size

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        raw_data = [[0] * self.seq_length +
            [self.vocab.get(w, 1) for w in line] +
            [2] * self.seq_length for line in lines]
        self.raw_data = raw_data #前后填充占位符
        # np.save(tensor_file, self.raw_data)

    def create_batches(self):
        xdata, ydata = list(), list()
        for row in self.raw_data:
            for ind in range(self.seq_length, len(row) - 1):
                xdata.append(row[ind-self.seq_length:ind])
                ydata.append(row[ind-self.seq_length+1:ind+1])
        self.num_batches = int(len(xdata) / self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.array(xdata[:self.num_batches * self.batch_size])
        ydata = np.array(ydata[:self.num_batches * self.batch_size])

        self.x_batches = np.split(xdata, self.num_batches, 0)
        self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, mini_frq=5):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.mini_frq = mini_frq #过滤出现次数少于min_frq的词

        input_file = os.path.join(data_dir, "input.txt") #输入文件
        vocab_file = os.path.join(data_dir, "vocab.pkl") #词汇文件
        tensor_file = os.path.join(data_dir, "data.npy") #原始数据

        self.preprocess(input_file, vocab_file, tensor_file) #预处理文件
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences):
        word_counts = collections.Counter()
        if not isinstance(sentences, list):
            sentences = [sentences]
        for sent in sentences:
            word_counts.update(sent)
        vocabulary_inv = ['<START>', '<UNK>', '<END>'] + \
                         [x[0] for x in word_counts.most_common() if x[1] >= self.mini_frq]
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            if lines[0][:1] == codecs.BOM_UTF8:
                lines[0] = lines[0][1:]
            lines = [line.strip().split() for line in lines]

        self.vocab, self.words = self.build_vocab(lines)
        self.vocab_size = len(self.words)
        #print 'word num: ', self.vocab_size

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        raw_data = [[0] * self.seq_length +
            [self.vocab.get(w, 1) for w in line] +
            [2] * self.seq_length for line in lines]
        self.raw_data = raw_data #前后填充占位符
        # np.save(tensor_file, self.raw_data)

    def create_batches(self):
        xdata, ydata = list(), list()
        for row in self.raw_data:
            for ind in range(self.seq_length, len(row)):
                xdata.append(row[ind-self.seq_length:ind])
                ydata.append([row[ind]])
        self.num_batches = int(len(xdata) / self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.array(xdata[:self.num_batches * self.batch_size])
        ydata = np.array(ydata[:self.num_batches * self.batch_size])

        self.x_batches = np.split(xdata, self.num_batches, 0)
        self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


def test():
    data_dir = '../data/xinhua'
    #data_dir = '../data/tinyshakespeare'
    batch_size = 64
    seq_length = 3
    loader = TextLoader2(data_dir, batch_size, seq_length)
    xdata, ydata = loader.next_batch()
    print len(xdata), len(ydata)
    print len(xdata[1]), len(ydata[1])
    print xdata[1]
    print ydata[1]
    for ind in xdata[1]:
        print loader.words[ind]
    for ind in ydata[1]:
        print loader.words[ind]
    #print [loader.words[ind] for ind in xdata[2]]
    #print [loader.words[ind] for ind in ydata[2]]

if __name__ == '__main__':
    test()