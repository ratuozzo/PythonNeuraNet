import keras
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from tensorflow import set_random_seed
import os
import csv
import pandas as pd
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def seedy(s):
    np.random.seed(s)
    set_random_seed(s)


class AutoEncoder:
    def __init__(self):

        self.train = pd.read_csv('train-bigDick.csv', nrows=128)

        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        num_hidden_1 = 256  # 1st layer num features
        num_hidden_2 = 128  # 2nd layer num features (the latent dim)
        self.num_input = 784  # MNIST data input (img shape: 28*28)

        self.placeholder = tf.placeholder("float", [None, self.num_input])
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, self.num_input])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_input])),
        }

    def _encoder(self,x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))

        return layer_2

    def _decoder(self,x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        return layer_2

    def encoder_decoder(self):

        ec = self._encoder(self.placeholder)
        dc = self._decoder(ec)

        return dc

    def get_batch(self,data, i, size):
        return data[range(i * size, (i + 1) * size)]

    def fit(self):

        num_steps = 30000
        batch_size = 128
        display_step = 1000



        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)
            # Training
            print(self.train)
            for i in range(1, num_steps + 1):
                for j in range(np.shape(self.train)[0] // batch_size):
                    batch_x, _ = self.get_batch(self.train, j, batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, l = sess.run([optimizer, loss], feed_dict={self.placeholder: batch_x})
                    # Display logs per step
                    if i % display_step == 0 or i == 1:
                        print('Step %i: Minibatch Loss: %f' % (i, l))





    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        else:
            self.encoder.save(r'./weights/encoder_weights.h5')
            self.decoder.save(r'./weights/decoder_weights.h5')
            self.model.save(r'./weights/ae_weights.h5')



if __name__ == '__main__':
    seedy(2)
    ae = AutoEncoder()

    prediction = ae.encoder_decoder()
    data = ae.placeholder

    loss = tf.reduce_mean(tf.pow(data - prediction, 2))
    optimizer = tf.train.RMSPropOptimizer(0.00023).minimize(loss)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    ae.fit()
    ae.save()
