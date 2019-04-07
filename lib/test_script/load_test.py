import tensorflow as tf 
import numpy as np 
from tensorflow.python import pywrap_tensorflow

weights_dir = '/home/vador/Documents/project/AI/drl-rpn-tf-video/pretrained-data/data3D/'

model = weights_dir + 'model_test'
meta_graph =  weights_dir + 'model_test.meta'

with tf.Session() as sess:
  saver = tf.train.import_meta_graph(model + '.meta')
  saver.restore(sess, model)
  for var in tf.global_variables():
    print(var.name)
    print(sess.run(var))
