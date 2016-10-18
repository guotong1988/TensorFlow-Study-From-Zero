# -*- coding:utf-8 -*-
'''
A Bidirectional Reccurent Neural Network (LSTM) implementation example using TensorFlow.
使用的数据来源(http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory的论文: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
'''

import tensorflow as tf

# 导入 MINST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)

'''
为了使用bidirectional reccurent neural network来做图片分类,
我们把一张图片考虑成 像素的序列 . 因为MNIST图片的大小是28*28px,
我们就相当于处理28个序列，每个序列 28 steps
'''

# 一些参数
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# 神经网络参数
n_input = 28 # MNIST data input (img shape: 28*28)      --28行      --同时感觉如果是28列，每列28个也行
n_steps = 28 # timesteps -- 感觉因为是RNN，所以会有这个概念  --每行28个
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

x = tf.placeholder("float", [None, n_steps, n_input])#设置成None？？？
y = tf.placeholder("float", [None, n_classes])

# 定义 W和b
weights = {
    # 使用 2*n_hidden 因为 foward + backward cells
    'hidden': tf.Variable(tf.random_normal([n_input, 2*n_hidden])),
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([2*n_hidden])),#这个好像没有用到
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#https://img3.doubanio.com/view/photo/photo/public/p2390342585.jpg
#这幅图取自97年的BRNN的原论文
#根据这幅图所示，这个函数才是实现了完整的 输入->Y序列 ，而不是一个tf.nn.bidirectional_rnn方法
#根据这幅图所示，所以tf.nn.bidirectional_rnn返回的output是256维
def BiRNN(x, weights, biases):

    # 准备输入X的shape以满足`bidirectional_rnn`函数的要求
    # 现在的X的shape: (batch_size, n_steps, n_input)
    # 根据tf.nn.bidirectional_rnn接口文档，X要求的shape: 长度为'n_steps'以及shape是(batch_size, n_input)的tensor的list

    # 将batch_size和n_steps转置
    x = tf.transpose(x, [1, 0, 2])
    # reshape为(n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # split成n_steps个(batch_size, n_input)的tensor的list
    x = tf.split(0, n_steps, x)#x此时已是一个长度为28的list，每个元素是（batch_size,28行）

    # 前向的 cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # 后向的 cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # 得到 lstm cell 的输出
    outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                            dtype=tf.float32)

    # 因为inputX是一个list，所以output也是一个长度n_steps的list ，感觉下面的W只是为了把维数搞到10，不过这个W应该是BRNN的一部分
    # 每个output是（batch_size,256） 矩阵乘以 W是（256×10）
    temp = tf.matmul(outputs[-1], weights['out'])
    temp = temp + biases['out']#加号左边是(batch_size,10),加号右边是(10)https://github.com/aymericdamien/TensorFlow-Examples/issues/75
    return temp

pred = BiRNN(x, weights, biases)#shape是（batch_size，10）和y一样

# 定义 loss 和 optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估模型时用的
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

# 启动 graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)#batch_x是(128,784),batch_y是(128,10)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # 计算 批量 accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # 计算 批量 loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    # 测试模型 计算 accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label})
