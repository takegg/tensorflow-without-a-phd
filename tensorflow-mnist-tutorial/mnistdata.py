# encoding: UTF-8
# Copyright 2018 Google.com
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
import numpy as np

import sys
sys.path.append("mlengine")
from trainer.task import load_mnist_data  # import from mlengine/trainer/task.py
from trainer.task import load_dataset     # import from mlengine/trainer/task.py

# 这会将整个数据集加载到内存中的numpy数组中
# 这使用tf.data.Dataset来避免重复代码。
# 通常，如果您已经有tf.data.Dataset，请加载
# 它对内存没用。 这里的目标是教育：
# 教授神经网络基础知识，而不必现在解释tf.data.Dataset。这个概念将在后面介绍。
# 使用tf.data.Dataset的正确方法是调用特征，labels = tf_dataset.make_one_shot_iterator().get_next（）然后直接在你的Tensorflow模型中使用“features”和“labels”。 这些tensorflow节点在执行时将自动触发下一批数据的加载。
# The sample that uses tf.data.Dataset correctly is in mlengine/trainer.

class MnistData(object): # 定义MnistData类，继承自object类，

    def __init__(self, tf_dataset, one_hot, reshape): # 定义类用的内部方法__init__，用于绑定对象属性，self将指向实例。
        self.pos = 0
        self.images = None
        self.labels = None
        # 通过10000个chunk将整个数据集加载到内存中
        tf_dataset = tf_dataset.batch(10000)
        tf_dataset = tf_dataset.repeat(1)
        features, labels = tf_dataset.make_one_shot_iterator().get_next()
        if not reshape:
            features = tf.reshape(features, [-1, 28, 28, 1])
        if one_hot:
            labels = tf.one_hot(labels, 10)
        with tf.Session() as sess:
            while True:
                try:
                    feats, labs = sess.run([features, labels])
                    self.images = feats if self.images is None else np.concatenate([self.images, feats])
                    self.labels = labs if self.labels is None else np.concatenate([self.labels, labs])
                except tf.errors.OutOfRangeError:
                    break


    def next_batch(self, batch_size):
        if self.pos+batch_size > len(self.images) or self.pos+batch_size > len(self.labels):
            self.pos = 0
        res = (self.images[self.pos:self.pos+batch_size], self.labels[self.pos:self.pos+batch_size])
        self.pos += batch_size
        return res


class Mnist(object): # Mnist类
    def __init__(self, train_dataset, test_dataset, one_hot, reshape):
        self.train = MnistData(train_dataset, one_hot, reshape) # MnistData的训练实例
        self.test = MnistData(test_dataset, one_hot, reshape) # MnistData的测试实例


def read_data_sets(data_dir, one_hot, reshape):
    train_images_file, train_labels_file, test_images_file, test_labels_file = load_mnist_data(data_dir)
    train_dataset = load_dataset(train_images_file, train_labels_file)
    train_dataset = train_dataset.shuffle(60000)
    test_dataset = load_dataset(test_images_file, test_labels_file)
    mnist = Mnist(train_dataset, test_dataset, one_hot, reshape)
    return mnist
