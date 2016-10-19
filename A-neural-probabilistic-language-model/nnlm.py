#encoding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#A neural probabilistic language model的tensorflow实现
from input_data import *

import numpy as np
import tensorflow as tf
import argparse
import time
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/xinhua',
                       help='data directory containing input.txt')
    parser.add_argument('--batch_size', type=int, default=120,
                       help='minibatch size')
    parser.add_argument('--win_size', type=int, default=5,
                       help='context sequence length')
    parser.add_argument('--hidden_num', type=int, default=256,
                       help='number of hidden layers')
    parser.add_argument('--word_dim', type=int, default=256,
                       help='number of word embedding')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--grad_clip', type=float, default=10.,
                       help='clip gradients at this value')

    args = parser.parse_args() #参数集合

    #准备训练数据
    data_loader = TextLoader(args.data_dir, args.batch_size, args.win_size)
    args.vocab_size = data_loader.vocab_size

    #准备测试例子
    test_words = ['<START>','<START>','<START>', '<START>','党中央']
    test_words_ids = [data_loader.vocab.get(w, 1) for w in test_words]

    #模型定义
    graph = tf.Graph()
    with graph.as_default():
        #输入变量
        input_data = tf.placeholder(tf.int32, [args.batch_size, args.win_size])
        targets = tf.placeholder(tf.int64, [args.batch_size, 1])
        test_words = tf.placeholder(tf.int64, [args.win_size])

        #模型参数
        with tf.variable_scope('nnlm' + 'embedding'):
            embeddings = tf.Variable(tf.random_uniform([args.vocab_size, args.word_dim], -1.0, 1.0))
            embeddings = tf.nn.l2_normalize(embeddings, 1)

        with tf.variable_scope('nnlm' + 'weight'):
            weight_h = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim + 1, args.hidden_num],
                            stddev=1.0 / math.sqrt(args.hidden_num)))#weight_h(1281,256)
            softmax_w = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim, args.vocab_size],
                            stddev=1.0 / math.sqrt(args.win_size * args.word_dim)))#softmax_w(1280,10000)
            softmax_u = tf.Variable(tf.truncated_normal([args.hidden_num + 1, args.vocab_size],
                            stddev=1.0 / math.sqrt(args.hidden_num)))#softmax_u(257,10000)


        #得到上下文的隐藏层表示
        def infer_output(input_data):
            inputs_emb = tf.nn.embedding_lookup(embeddings, input_data)#embeddings(10000,256),input_data(120,5)
            inputs_emb = tf.reshape(inputs_emb, [-1, args.win_size * args.word_dim]) #inputs_emb(120,5,256)->(120,1280)
            temp = tf.pack([tf.shape(input_data)[0], 1]) #temp(2)
            temp2 = tf.ones(temp)#temp2(120,1)
            inputs_emb_add = tf.concat(1, [inputs_emb, temp2])#inputs_emb_add(120,1281)

            inputs = tf.tanh(tf.matmul(inputs_emb_add, weight_h))#inputs(120,256)
            inputs_add = tf.concat(1, [inputs, tf.ones(tf.pack([tf.shape(input_data)[0], 1]))])#inputs_add(120,257)
            outputs = tf.matmul(inputs_add, softmax_u) + tf.matmul(inputs_emb, softmax_w)#outputs(120,10000)
            outputs = tf.clip_by_value(outputs, 0.0, args.grad_clip)#outputs(120,10000)
            outputs = tf.nn.softmax(outputs)
            return outputs

        outputs = infer_output(input_data)
        one_hot_targets = tf.one_hot(tf.squeeze(targets), args.vocab_size, 1.0, 0.0)

        loss = -tf.reduce_mean(tf.reduce_sum(tf.log(outputs) * one_hot_targets, 1))
        optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)

        #输出词向量
        test_outputs = infer_output(tf.expand_dims(test_words_ids, 0))
        test_outputs = tf.arg_max(test_outputs, 1)

        embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / embeddings_norm

    #模型训练
    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        for e in range(args.num_epochs):
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {input_data: x, targets: y}
                train_loss,  _ = sess.run([loss, optimizer], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" .format(
                        b, data_loader.num_batches,
                        e, train_loss, end - start))
                print(normalized_embeddings.eval()[0])
            np.save('nnlm_word_embeddings', normalized_embeddings.eval())

        #样例测试
        feed = {test_words : test_words_ids}
        test_outputs, _ = sess.run([test_outputs], feed)
        print '>'.join(test_words)
        print data_loader.words[test_outputs[0]]

if __name__ == '__main__':
    main()