import tensorflow as tf
from nets.vgg16 import vgg16
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
from model.test import test_net
import os

# load model for detection
net = vgg16()
net.create_architecture("TEST", tag='default')
net.build_drl_rpn_network(False)

# load database of images
imdb_name = 'voc_2007_test'
imdb = get_imdb(imdb_name)

# Set class names in config file based on IMDB
class_names = imdb.classes
cfg_from_list(['CLASS_NAMES', [class_names]])

# Update config depending on if class-specific history used or not
cfg_from_list(['DRL_RPN.USE_POST', False]) # THROWS AN ERROR IF SET TO TRUE, WHY ?

# Specify if run drl-RPN in auto mode or a fix number of iterations
cfg_from_list(['DRL_RPN_TEST.NBR_FIX', 0]) 

# Specify if run drl-RPN in auto mode or a fix number of iterations
cfg_from_list(['DIMS_TIME', 4])

# test the network
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True

#model = None
model = '/home/vador/Documents/project/AI/drl-rpn-tf-video/output-weights/voc_2007_train/output/default/voc_2007_train/vgg16_drl_rpn_iter_1.ckpt'
#model = '/home/vador/Documents/project/AI/drl-rpn-tf-video/drl-rpn-voc2007-2012-trainval/vgg16_drl_rpn_iter_110000.ckpt'
weight = 'weights3d'

if model:
    filename = os.path.splitext(os.path.basename(model))[0]
else:
    filename = weight

with tf.Session(config=tfconfig) as sess:
    # load model
    if model:
        print(('Loading model check point from {:s}').format(model))
        # Why the following line is needed ?? if i remove it, some var are not initialized ! WHY ?
        #sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver() #import_meta_graph(model + '.meta')
        saver.restore(sess, model)
        for var in tf.global_variables():
            if var.name == 'h_relu_weights_video:0':
                print(var.name)
                print(var.shape)

    else:
        print(('Loading initial weights from {:s}').format(weight))
        sess.run(tf.global_variables_initializer())
        print('Loaded.')

    test_net(sess, net, imdb, filename)



