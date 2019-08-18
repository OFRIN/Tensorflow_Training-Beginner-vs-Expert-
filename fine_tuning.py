import numpy as np
import tensorflow as tf

import vgg_16.VGG16 as vgg

from Define import *

def Global_Average_Pooling(x):    
    pool_size = np.shape(x)[1:3][::-1]
    return tf.layers.average_pooling2d(inputs = x, pool_size = pool_size, strides = 1)

def fine_tuning(input_var, is_training):
    x = input_var - VGG_MEAN

    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        x = vgg.vgg_16(x, num_classes=1000, is_training=is_training, dropout_keep_prob=0.5)
    
    x = Global_Average_Pooling(x)
    x = tf.contrib.layers.flatten(x)

    logits = tf.layers.dense(x, CLASSES, use_bias = False, name = 'logits')
    predictions = tf.nn.softmax(logits)

    return logits, predictions
