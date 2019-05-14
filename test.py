from keras.models import load_model
import numpy as np
from keras import models
import tensorflow as tf
import math

sess = tf.Session()
saver = tf.train.import_meta_graph('/weights/model.ckpt.meta')
saver.restore(sess, "model")

inputs = np.array([[400,450,500,900,1000,758,544,867,1,2]])
sess.run(saver, feed_dict=[400,450,500,900,1000,758,544,867,1,2])

print('Input: {}'.format(inputs))
