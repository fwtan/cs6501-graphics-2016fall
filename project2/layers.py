#!/usr/bin/env python

import math, re
import numpy as np
import tensorflow as tf

class batch_normalization(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, scope, epsilon=1e-5, momentum=0.9):
        self.scope    = scope
        self.epsilon  = epsilon
        self.momentum = momentum
        self.ema      = tf.train.ExponentialMovingAverage(decay=self.momentum)

    def __call__(self, x, trainable=True):
        shape = x.get_shape().as_list()

        if trainable:
            with tf.variable_scope(self.scope):
                # assuming x of shape [batch_size, height, width, feature_channel] or [batch_size, feature_channel]
                self.beta  = tf.get_variable("beta",  [shape[-1]], initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.random_normal_initializer(1., 0.02))

                try:
                    # spatial batchnorm for conv layers
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    # simple batchnorm for fully connected layers
                    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')

                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        return tf.nn.batch_normalization(x, mean, var, self.beta, self.gamma, self.epsilon)

def linear(x, output_dim, scope='linear'):
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope):
        weights = tf.get_variable(name='weights',
                                shape=[shape[-1], output_dim],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0)/math.sqrt(float(shape[-1]))))
        biases = tf.get_variable(name='biases',
                            shape=[output_dim],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))

        return tf.nn.bias_add(tf.matmul(x, weights), biases)

def conv2d(x, output_dim,
           k_h=3, k_w=3,
           d_h=1, d_w=1,
           padding='SAME',
           scope='conv2d'):

    with tf.variable_scope(scope):
        w = tf.get_variable('w', [k_h, k_w, x.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.001))
        conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

def deconv2d(x, output_shape,
             k_h, k_w, d_h, d_w,
             padding='SAME',
             scope="deconv2d"):

    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.001))
        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def leaky_relu(x, leak=0.2, name=None):
    return tf.maximum(x, leak * x, name=name)
