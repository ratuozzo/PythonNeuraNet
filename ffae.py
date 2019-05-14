import keras
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from tensorflow import set_random_seed
import os
import csv
from numpy import genfromtxt
import pandas as pd
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow as tf
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


def seedy(s):
    np.random.seed(s)
    set_random_seed(s)


class AutoEncoder:
    def __init__(self):

        self.cost_summary = None
        self.train = genfromtxt('train.csv', delimiter=',')
        self.test = genfromtxt('test.csv', delimiter=',')

        self.train = X = scale( self.train, axis=0, with_mean=True, with_std=True, copy=True )

        num_hidden_1 = 8  # 1st layer num features
        num_hidden_2 = 6  # 2nd layer num features (the latent dim)
        num_hidden_3 = 4  # 2nd layer num features (the latent dim)
        self.num_input = 10  # MNIST data input (img shape: 28*28)
        self.learning_rate = 0.0001

        self.placeholder = tf.placeholder("float", [None, self.num_input], name="Input")
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
            'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3])),
            'decoder_h1': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2])),
            'decoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
            'decoder_h3': tf.Variable(tf.random_normal([num_hidden_1, self.num_input])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
            'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
            'decoder_b1': tf.Variable(tf.random_normal([num_hidden_2])),
            'decoder_b2': tf.Variable(tf.random_normal([num_hidden_1])),
            'decoder_b3': tf.Variable(tf.random_normal([self.num_input])),
        }


        #self.prediction = self.encoder_decoder()
        #self.data = self.placeholder
        #self.loss = tf.reduce_mean(tf.pow(self.data - self.prediction, 2))
        #self.error = tf.identity(tf.pow(self.data - self.prediction, 2))
        #self.optimizer = tf.train.RMSPropOptimizer(0.0003).minimize(self.loss)
        #self.sess = tf.Session()

        number_of_neurons_first_layer = 6
        number_of_neurons_second_layer = 2

        # model (diabolo network)
        We1 = tf.Variable(tf.random_normal([self.num_input, number_of_neurons_first_layer], dtype=tf.float32))
        be1 = tf.Variable(tf.zeros([number_of_neurons_first_layer]))

        We2 = tf.Variable(
            tf.random_normal([number_of_neurons_first_layer, number_of_neurons_second_layer], dtype=tf.float32))
        be2 = tf.Variable(tf.zeros([number_of_neurons_second_layer]))

        Wd1 = tf.Variable(
            tf.random_normal([number_of_neurons_second_layer, number_of_neurons_first_layer], dtype=tf.float32))
        bd1 = tf.Variable(tf.zeros([number_of_neurons_first_layer]))

        Wd2 = tf.Variable(tf.random_normal([number_of_neurons_first_layer, self.num_input], dtype=tf.float32))
        bd2 = tf.Variable(tf.zeros([self.num_input]))

        self.new_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.num_input])

        encoding = tf.nn.tanh(tf.add(tf.matmul(self.new_placeholder, We1), be1))
        encoding = tf.nn.leaky_relu(tf.add(tf.matmul(encoding, We2), be2))
        decoding = tf.nn.tanh(tf.add(tf.matmul(encoding, Wd1), bd1))
        self.decoded = tf.nn.leaky_relu(tf.add(tf.matmul(decoding, Wd2), bd2),name='model')

        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.new_placeholder, self.decoded)))
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.batch_size = 128

    def get_batch(self, data, i, size):
        return np.array(data[i:i+size])

    def fit(self):

        tf.set_random_seed(1)

        saver = tf.train.Saver()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        loss_train = []

        num_steps = 6000


        for i in range(1, num_steps + 1):
            for j in range(np.shape(self.train)[0] // self.batch_size):
                batch_data = self.get_batch(self.train, j, self.batch_size)
                sess.run(self.train_step, feed_dict={self.new_placeholder: batch_data})
            lt = sess.run(self.loss, feed_dict={self.new_placeholder: self.train})
            loss_train.append(lt)
            if i % 50 == 0 or i == num_steps - 1:
                print('iteration {0}: loss train = {1:.4f}'.format(i, lt))

        save_path = saver.save(sess, '/weights/model.ckpt')
        print("Model saved in path: %s" % save_path)

        plt.figure()
        plt.clf()
        plt.cla()
        plt.semilogy(loss_train)
        plt.ylabel('loss')
        plt.xlabel('Iterations')
        plt.legend(['train'], loc='upper right')
        plt.show()

        output = sess.run(self.decoded, feed_dict={self.new_placeholder: self.train})

        rows = output.shape[0]
        cols = output.shape[1]
        print(rows)
        print(cols)

        for x in range(0, rows - 1):
                print("Train ",x,":",self.train[x])
                print("Output",x,":",output[x])


if __name__ == '__main__':
    seedy(2)
    ae = AutoEncoder()
    ae.fit()


