# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflowvisu
import math
import mnistdata
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# 5层神经网络
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (relu)         W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (relu)         W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (relu)         W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (relu)         W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
# 下载图片和标签到mnist.test(10k图片+标签)和mnist.train(60k图片+标签)
mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

# 输入X：28乘28灰度图，第一维(None)将索引mini-batch中的图片。
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# 正确答案将在这
Y_ = tf.placeholder(tf.float32, [None, 10])
# 学习速率可变的step
step = tf.placeholder(tf.int32)

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
# 5层和神经元的数量(最后一次后10个softmax神经元)
L = 200
M = 100
N = 60
O = 30
# 权重初始化为-0.2到+0.2之间的随机小值
# 当使用RELUs时，确保biases初始化为小*正*值，例如0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O])/10)
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# The model
XX = tf.reshape(X, [-1, 784])
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# 交叉熵损失函数(= -sum(Y_i * log(Yi)) )，标准化为100个图片的批量。
# TensorFlow提供softmax_cross_entropy_with_logits函数以避免数值稳定性。
# NaN的问题用log(0)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# 训练模型的精确度在0(最坏)和1(最好)之间
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib可视化
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis()

# training step
# 学习速率是：0.0001 + 0.003 * (1/e)^(step/2000))，即指数衰减从0.003到0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# 初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# 你可以在训练模型的循环中调用此函数，每次100张图
def training_step(i, update_test_data, update_train_data):

    # 在100张带100个标签的图的批上训练
    batch_X, batch_Y = mnist.train.next_batch(100)

    # 计算可视化的训练值
    if update_train_data:
        a, c, im, w, b, l = sess.run([accuracy, cross_entropy, I, allweights, allbiases, lr],
                                     feed_dict={X: batch_X, Y_: batch_Y, step: i})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(l) + ")")
        datavis.append_training_curves_data(i, a, c)
        datavis.update_image1(im)
        datavis.append_data_histograms(i, w, b)

    # 计算可视化的测试值
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)

    # 反向传播training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, step: i})

datavis.animate(training_step, iterations=10000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start=True)

# 保存动画为视频，添加save_movie=True作为datavis.animate的一个参数
# 使用以下行代替datavis.animate行以禁止可视化
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# 一些预期结果：
# (在运行中，如果使用sigmoids，所有biasess初始化为0，如果使用RELUs，除去初始化为0的最后一个所有biases初始化为0.1)

## 学习速率为0.003,10k的迭代
# 最终测试准确率为0.9788(sigmoid-缓慢开始，训练cross-entropy在结束时不稳定)
# 最终测试准确率为0.9825（relu-在第一次1500个迭代中在0.97以上，但曲线嘈杂）

## 现在已0.0001的学习速率，10k的迭代
# 最终测试准确率为0.9722(relu-慢但是曲线平滑，在20k的迭代中将更高)

## 学习速率衰减从0.003到0.0001，衰减速率为2000,10k迭代。
# 最终测试准确率为0.9746(sigmoid-训练交叉熵不稳定)
# final test accuracy = 0.9824 (relu - training set fully learned, test accuracy stable)
# 最终测试准确率为0.9824(relu-训练集充分学习，测试集准确率稳定)