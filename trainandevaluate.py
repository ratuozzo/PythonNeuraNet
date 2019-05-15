import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn import preprocessing
import numpy as np
from tensorflow import set_random_seed
import os
import csv
import sys
from numpy import genfromtxt
import pandas as pd
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow as tf
from sklearn.preprocessing import scale
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



def seedy(s):
    np.random.seed(s)
    set_random_seed(s)


class AutoEncoder:
    def __init__(self):

        self.max_error = -1
        self.cost_summary = None
        self.train = genfromtxt(sys.argv[1], delimiter=',')
        self.test = genfromtxt('evaluate.csv', delimiter=',')

        self.train = preprocessing.scale(self.train)
        self.test = preprocessing.scale(self.test)

        self.anomalies = 0
        self.non_anomalies = 0

        self.non_anomalies_indexes = []

        self.learning_rate = float(sys.argv[2])
        print("Learning rate: ", self.learning_rate)
        
        self.num_input = 10  
        number_of_neurons_first_layer = 6
        number_of_neurons_second_layer = 2

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

        self.new_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.num_input],name='model')

        encoding = tf.nn.tanh(tf.add(tf.matmul(self.new_placeholder, We1), be1))
        encoding = tf.nn.leaky_relu(tf.add(tf.matmul(encoding, We2), be2))
        decoding = tf.nn.tanh(tf.add(tf.matmul(encoding, Wd1), bd1))
        self.decoded = tf.nn.leaky_relu(tf.add(tf.matmul(decoding, Wd2), bd2))

        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.new_placeholder, self.decoded)))
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.batch_size = int(sys.argv[3])

    def get_batch(self, data, i, size):
        return np.array(data[i:i+size])

    def fit(self):

        tf.set_random_seed(1)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        loss_train = []

        num_steps = int(sys.argv[4])

        for i in range(1, num_steps + 1):
            for j in range(np.shape(self.train)[0] // self.batch_size):
                batch_data = self.get_batch(self.train, j, self.batch_size)
                sess.run(self.train_step, feed_dict={self.new_placeholder: batch_data})
            lt = sess.run(self.loss, feed_dict={self.new_placeholder: self.train})
            loss_train.append(lt)
            if i % 50 == 0 or i == num_steps - 1:
                print('iteration {0}: loss train = {1:.5f}'.format(i, lt))

        plt.figure()
        plt.clf()
        plt.cla()
        plt.semilogy(loss_train)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.legend(['Training'], loc='upper right')
        plt.savefig('plots/lossvsiteration.png')
        plt.clf()
        plt.cla()

        train_output = sess.run(self.decoded, feed_dict={self.new_placeholder: self.train})
        test_output = sess.run(self.decoded, feed_dict={self.new_placeholder: self.test})

        rows = train_output.shape[0]
        cols = train_output.shape[1]

        for x in range(0, rows - 1):
            sum = 0
            for y in range(0,9):
                sum = sum + (self.train[x][y] - train_output[x][y])**2
            error = math.sqrt(sum/10)
            if self.max_error < error:
                self.max_error = error
        print("Max Error", self.max_error)

        self.generate_scatter_plot(self.train, train_output,'train')
        self.generate_scatter_plot(self.test, test_output,'evaluate')

        print("Anomalies: ", self.anomalies,
              (self.anomalies /(self.anomalies+self.non_anomalies))*100," %")
        print("Non Anomalies: ", self.non_anomalies,
              (self.non_anomalies /(self.anomalies+self.non_anomalies))*100," %")
        print("NonAnomalies Indexes: ",self.non_anomalies_indexes)


    def generate_scatter_plot(self, input, output, type):
        self.non_anomalies = 0
        self.anomalies = 0
        for x in range(0, output.shape[0] - 1):
            sum = 0
            for y in range(0, 9):
                sum = sum + (input[x][y] - output[x][y]) ** 2
            error = math.sqrt(sum / 10)
            print("Plotting point ", x, " of ", output.shape[0] - 1)
            if error <= self.max_error:
                plt.plot(x, error, 'o', color='green')
                self.non_anomalies = self.non_anomalies + 1
                self.non_anomalies_indexes.append(x)
            else:
                plt.plot(x, error, 'o', color='red')
                self.anomalies = self.anomalies + 1


        plt.ylabel('Prediction MSE')
        plt.xlabel('Data Entry')
        anomaly = mpatches.Patch(color='red', label='Anomaly')
        non_anomaly = mpatches.Patch(color='green', label='Non Anomaly')
        plt.legend(handles=[anomaly,non_anomaly], loc='upper right')
        plt.savefig('plots/'+type+'.png')
        plt.clf()
        plt.cla()
       

if __name__ == '__main__':
    seedy(2)
    ae = AutoEncoder()
    ae.fit()


