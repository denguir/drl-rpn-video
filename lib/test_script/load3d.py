import tensorflow as tf 
import numpy as np 
from tensorflow.python import pywrap_tensorflow

weights3d_dir = '/home/vador/Documents/project/AI/drl-rpn-tf-video/output-weights/drl-rpn-paris/output/vgg16_drl_rpn/paris_train/'
#weights3d_dir = '/home/vador/Documents/project/AI/drl-rpn-tf/data/pre-trained/data3D/drl-rpn-voc2007-2012-trainval/'
model3d = weights3d_dir + 'vgg16_drl_rpn_iter_0.ckpt'
meta3d = weights3d_dir + 'vgg16_drl_rpn_iter_0.ckpt.meta'


def get_variables_in_checkpoint_file(file_name):
    try:
      reader = pywrap_tensorflow.NewCheckpointReader(file_name)
      var_to_shape_map = reader.get_variable_to_shape_map()
      return var_to_shape_map 
    except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(meta3d)
    saver.restore(sess, model3d)
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())

    weight = graph.get_tensor_by_name('xr_weights_base_video' + ":0")
    weight_np = sess.run(weight)
    #verify if the two depth dimension are equal
    print(weight_np[0,:,:,:,:] == weight_np[1,:,:,:,:])

    # variables = tf.global_variables()
    # Initialize all variables first
    # sess.run(tf.variables_initializer(variables, name='init'))
    # var_keep_dic = get_variables_in_checkpoint_file(model3d)
    # for name in var_keep_dic:
    #   print(name)