import tensorflow as tf 
import numpy as np 

weights3d_dir = '/home/vador/Documents/project/AI/drl-rpn-tf-video/pretrained-data/data3D/'
model3d = weights3d_dir + 'model_test'

vn1 = np.array(np.random.randint(0, 10, (4,5,5)))
vt1 = tf.Variable(vn1, name='v1')

vn2 = np.array(np.random.randint(0, 10, (4,5,5)))
vt2 = tf.Variable(vn2, name='v2')

vt3 = tf.multiply(vt1, vt2)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, model3d)


