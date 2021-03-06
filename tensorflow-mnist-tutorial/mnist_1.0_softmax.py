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

# 本文件必须在根目录才能运行.
import tensorflow as tf
import tensorflowvisu # 可视化(自定义模块)
import mnistdata # mnist数据集(自定义模块)
import math # 数学方法
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# 一层具有10个softmax神经元的神经网络
#
# · · · · · · · · · ·       (输入数据, 扁平化像素)                X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# 模型:
#
# Y = softmax( X * W + b)
#              X: 一百张28*28个像素的灰度图的矩阵，扁平化的（这一百张图片在一个小批量中）
#              W: 784行10列的权重矩阵
#              b: 10维的bias矢量
#              +： 添加一个广播：添加矢量到矩阵的每行（numpy）
#              softmax(matrix) 每行都应用softmax
#              softmax(line) 对每个值应用指数，然后除以结果行的范数
#              Y: 输出一百行十列的矩阵

# 下载图片、标签到测试集（10k的图片和标签）和训练集（60k图片和标签）
mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)


# 输入X：28*28的灰度图，第一维（None）将索引在小批量中的图片。形状数组代表每个维度元素数量
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) #表示None×28行×28列×1个颜色个值
# 正确标签将在这
Y_ = tf.placeholder(tf.float32, [None, 10])
# 权重 W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([784, 10]))
# 偏量 b[10]
b = tf.Variable(tf.zeros([10]))

# 扁平化图片到一个单行像素
# -1 在形状定义中表示“保留元素数量的唯一可能维度”。实践中，它将是小批量的图像数。
XX = tf.reshape(X, [-1, 784])

# 模型
Y = tf.nn.softmax(tf.matmul(XX, W) + b)
# 笔记：Y是784行10列的矩阵，每行为每张图10个数字的概率。

# 损失函数: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y:计算后输出向量(预测标签)
#                           Y_：所需要输出的向量(正确标签)

# 交叉熵
# 以log获取每个元素的log，以元素乘以张量元素
# reduce_mean（减少平均值）将添加张量中的所有组件
# 所以这里我们得到在批量中所有图片的总交叉熵
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # 一批一百张图的标准化
                                                          # 乘10因为平均值包含一个不需要的除以10

# 训练模型的精确度介于0（坏）和1（好）之间
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)) #正确预测(根据每个元素是否相等返回一个布尔向量)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #精确度(整体平均值)

# 训练中，学习速率=0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
# 笔记，用梯度下降优化器来求最小交叉熵。

# matplotlib可视化
allweights = tf.reshape(W, [-1])
allbiases = tf.reshape(b, [-1])
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = tensorflowvisu.MnistDataVis()

# 初始化
init = tf.global_variables_initializer() #笔记：初始化全局变量并返回
sess = tf.Session() # 笔记：生成会话。
sess.run(init) # 笔记：开始运行，主要代码到此结束。


# 你可以在循环中调用此函数以训练模型，一次一百张图
def training_step(i, update_test_data, update_train_data):

    # 对一百张图带一百个标签的批次进行训练
    batch_X, batch_Y = mnist.train.next_batch(100) # batch_X：图片，batch_Y：标签

    # 计算可视化的训练值
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # 计算可视化的测试值
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # 反向传播训练步骤
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})


datavis.animate(training_step, iterations=2000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)

# 将动画保存为影片，将save_movie=True添加为datavis.animate的参数。
# 要禁止可视化使用以下行，而不是datavis.animate行。
# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# 最终最大测试精度=0.9268（10k次迭代）。在前两千次迭代中精确度达到0.92峰值。
