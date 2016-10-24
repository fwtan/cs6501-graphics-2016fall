#!/usr/bin/env python

import os, sys
import time, math
import cv2, cPickle
from imdb import imdb
from layers import *
from glob import glob
import numpy as np
import tensorflow as tf

class colorization(object):
    """docstring for colorization."""
    def __init__(self, sess, db, batch_size, train_shape, output_dir):
        super(colorization, self).__init__()
        self.sess        = sess
        self.batch_size  = batch_size
        self.train_shape = train_shape
        self.train_roidb = db.train_roidb
        self.val_roidb   = db.val_roidb
        self.output_dir  = output_dir
        self.bn = {}

        """Randomly permute the training roidb."""
        self.train_perm  = np.random.permutation(np.arange(len(self.train_roidb)))
        self.val_perm    = np.random.permutation(np.arange(len(self.val_roidb)))

        self.train_cur = 0
        self.val_cur   = 0

        # batch_normalization
        for i in xrange(1,5):
            bn_name = 'cbn%d'%i
            self.bn['f_cbn%d'%i] = batch_normalization(scope=bn_name)
        for i in xrange(4, 0, -1):
            bn_name = 'dbn%d'%i
            self.bn['f_dbn%d'%i] = batch_normalization(scope=bn_name)
        for i in xrange(1,8):
            bn_name = 'bn%d'%i
            self.bn['m_bn%d'%i] = batch_normalization(scope=bn_name)

        self.init_model();

    def init_model(self):
        self.input       = tf.placeholder(tf.float32, [self.batch_size, self.train_shape[0], self.train_shape[1], 1], name='input')
        self.full_target = tf.placeholder(tf.float32, [self.batch_size, self.train_shape[0], self.train_shape[1], 2], name='full_target')
        self.mean_target = tf.placeholder(tf.float32, [self.batch_size,                                           2], name='mean_target')

        self.full_inferred = self.full_inference(self.input, True)
        self.full_sample   = self.full_inference(self.input, False)

        self.mean_inferred = self.mean_inference(self.input, True)
        self.mean_sample   = self.mean_inference(self.input, False)

        self.full_inferred_loss = tf.div(tf.nn.l2_loss(tf.sub(self.full_inferred, self.full_target)), self.batch_size)
        self.mean_inferred_loss = tf.div(tf.nn.l2_loss(tf.sub(self.mean_inferred, self.mean_target)), self.batch_size)

        self.full_sample_loss = tf.div(tf.nn.l2_loss(tf.sub(self.full_sample, self.full_target)), self.batch_size)
        self.mean_sample_loss = tf.div(tf.nn.l2_loss(tf.sub(self.mean_sample, self.mean_target)), self.batch_size)

        self.full_inferred_loss_summary = tf.scalar_summary("full_inferred_loss", self.full_inferred_loss)
        self.mean_inferred_loss_summary = tf.scalar_summary("mean_inferred_loss", self.mean_inferred_loss)

        self.test_loss         = tf.Variable(0.0, trainable=False)
        self.test_loss_summary = tf.scalar_summary("test_loss", self.test_loss)

        self.t_vars = tf.trainable_variables()
        self.full_vars = [var for var in self.t_vars if 'full' in var.name]
        self.mean_vars = [var for var in self.t_vars if 'mean' in var.name]

        ##########################################################
        # checkpoint IO
        self.saver = tf.train.Saver()

    def load_checkpoint(self, model):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            return False
        model_dir = model
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            return False
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def save_checkpoint(self, model, step):
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_name = '%s.model'%model
        model_dir  = model
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def get_minibatch(self, mode='full', train=True, vis=False):
        #######################################################################
        if vis:
            output_dir = os.path.join(self.output_dir, 'minibatch')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        #######################################################################

        input_batch      = np.zeros((self.batch_size, self.train_shape[0], self.train_shape[1], 1), dtype=np.float32)
        if mode == 'full':
            target_batch = np.zeros((self.batch_size, self.train_shape[0], self.train_shape[1], 2), dtype=np.float32)
        else:
            target_batch = np.zeros((self.batch_size, 2), dtype=np.float32)

        if train:
            roidb = self.train_roidb
            perm  = self.train_perm
            cur   = self.train_cur
        else:
            roidb = self.val_roidb
            perm  = self.val_perm
            cur   = self.val_cur

        # permute
        if cur + self.batch_size >= len(roidb):
            perm = np.random.permutation(np.arange(len(roidb)))
            cur = 0

        db_inds = perm[cur : cur + self.batch_size]
        cur += self.batch_size
        #######################################################################

        for i in xrange(self.batch_size):
            rois    = roidb[db_inds[i]]
            im_path = rois['image']
            im_name, im_ext = os.path.splitext(os.path.basename(im_path))

            im = cv2.imread(im_path, cv2.IMREAD_COLOR)
            if rois['flipped']:
                im = cv2.flip(im, 1)
            im = im * rois['scale']
            im = cv2.resize(im, (self.train_shape[1], self.train_shape[0]))
            lab_im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            l_channel,a_channel,b_channel = cv2.split(lab_im)

            input_batch[i,:,:,:] = (l_channel - 128.0)
            if mode == 'full':
                target_batch[i,:,:,0] = (a_channel - 128.0)/128.0
                target_batch[i,:,:,1] = (b_channel - 128.0)/128.0
            else:
                target_batch[i,:,:,0] = (np.mean(a_channel) - 128.0)/128.0
                target_batch[i,:,:,1] = (np.mean(b_channel) - 128.0)/128.0

            #######################################################################
            if vis:
                output_path = os.path.join(output_dir, '%04d_'%i + im_name + im_ext)
                cv2.imwrite(output_path, im)
            #######################################################################
        return input_batch, target_batch

    def full_inference(self, input_batch, train):

        input_shape = input_batch.get_shape().as_list()

        with tf.variable_scope('full', reuse=(not train)):
            a = input_batch
            for i in xrange(1, 5):
                a = conv2d(a, 32 * (2**i), k_h=3, k_w=3, d_h=2, d_w=2, scope="conv%d"%i)
                a = self.bn['f_cbn%d'%i](a, trainable=train)
                a = leaky_relu(a)

            for i in xrange(4, 1, -1):
                a = deconv2d(a, [self.batch_size, int(input_shape[1]/(2**(i-1))), int(input_shape[2]/(2**(i-1))), 32 * (2**(i-1))],
                                k_h=3, k_w=3, d_h=2, d_w=2,
                                scope="deconv%d"%i)
                a = self.bn['f_dbn%d'%i](a, trainable=train)
                a = leaky_relu(a)

            a = deconv2d(a, [self.batch_size, input_shape[1], input_shape[2], 2],
                            k_h=3, k_w=3, d_h=2, d_w=2,
                            scope="deconv_out")

            return tf.tanh(a)

    def mean_inference(self, input_batch, train):
        input_shape = input_batch.get_shape().as_list()

        with tf.variable_scope('mean', reuse=(not train)):

            # conv1
            a = conv2d(input_batch, 96, k_h=7, k_w=7, d_h=2, d_w=2, scope='conv1')
            a = self.bn['m_bn1'](a, trainable=train)
            a = leaky_relu(a)
            a = tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            # conv2
            a = conv2d(a, 256, k_h=5, k_w=5, d_h=2, d_w=2, scope='conv2')
            a = self.bn['m_bn2'](a, trainable=train)
            a = leaky_relu(a)
            a = tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            # conv3
            a = conv2d(a, 384, k_h=3, k_w=3, d_h=1, d_w=1, scope='conv3')
            a = self.bn['m_bn3'](a, trainable=train)
            a = leaky_relu(a)
            a = tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            # conv4
            a = conv2d(a, 384, k_h=3, k_w=3, d_h=1, d_w=1, scope='conv4')
            a = self.bn['m_bn4'](a, trainable=train)
            a = leaky_relu(a)
            a = tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            # conv5
            a = conv2d(a, 256, k_h=3, k_w=3, d_h=1, d_w=1, scope='conv5')
            a = self.bn['m_bn5'](a, trainable=train)
            a = leaky_relu(a)
            a = tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            # linear1
            batch_size = a.get_shape().as_list()[0]
            a = linear(tf.reshape(a, [batch_size,-1]), 512, scope='fc6')
            a = self.bn['m_bn6'](a, trainable=train)
            a = leaky_relu(a)

            # linear2
            a = linear(a, 64, scope='fc7')
            a = self.bn['m_bn7'](a, trainable=train)
            a = leaky_relu(a)

            a = linear(a, 2, scope='fc_out')

            return tf.tanh(a)

    def train(self, num_iterations = 100000, mode='full'):
        # Optimizer
        if mode == 'full':
            train_vars  = self.full_vars
            train_loss  = self.full_inferred_loss
            train_summ  = self.full_inferred_loss_summary
            target      = self.full_target
            sample_loss = self.full_sample_loss
        else:
            train_vars  = self.mean_vars
            train_loss  = self.mean_inferred_loss
            train_summ  = self.mean_inferred_loss_summary
            target      = self.mean_target
            sample_loss = self.mean_sample_loss

        ###########################################################################
        # test accuracy temp vars
        test_loss_tmp = tf.Variable(0.0, trainable=False)
        # tensor for test iterations and zero
        test_iters = int(len(self.val_roidb)/self.batch_size)
        tf_iters = tf.constant(test_iters, dtype=tf.float32)
        tf_zero  = tf.constant(0.0, dtype=tf.float32)
        ###########################################################################

        optim = tf.train.AdamOptimizer(0.001, beta1=0.9).minimize(train_loss, var_list=train_vars)
        #init = tf.initialize_variables(var_list=train_vars)
        init = tf.initialize_all_variables()
        self.sess.run(init)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        log_dir = os.path.join(self.output_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.summary_writer = tf.train.SummaryWriter(log_dir, self.sess.graph)

        # Load checkpoint if available
        if self.load_checkpoint(model=mode):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Main loop
        for step in xrange(num_iterations):
            start_time = time.time()

            input_batch, target_batch = self.get_minibatch(mode=mode, train=True)
            feed_dict={ self.input: input_batch, target: target_batch }
            _, loss = self.sess.run([optim, train_loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if (step > 1 and step % 20 == 0) or (step + 1) == num_iterations:
                # Print status to stdout.
                print('Step %d: loss = %.4f (%.3f sec)' % (step, loss, duration))
                summary_str = self.sess.run(train_summ, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, step)
                self.summary_writer.flush()

            # Save a checkpoint.
            if (step > 1 and (step + 1) % 1000 == 0) or (step + 1) == num_iterations:
                self.save_checkpoint(mode, step)

            if mode == 'full':
                # Draw samples
                if (step > 1 and (step + 1) % 1000 == 0) or (step + 1) == num_iterations:
                    samples, loss = self.sess.run([self.full_sample, self.full_sample_loss], feed_dict=feed_dict)
                    self.draw_samples(input_batch, samples, target_batch, step)
                    print("[Sample] loss: %.8f" % loss)

            # Test
            if (step > 1 and (step + 1) % 5000 == 0) or (step + 1) == num_iterations:
                self.val_perm = np.random.permutation(np.arange(len(self.val_roidb)))
                self.val_cur = 0

                self.sess.run(tf.assign(test_loss_tmp, tf_zero))

                for k in xrange(test_iters):
                    input_batch, target_batch = self.get_minibatch(mode=mode, train=False)
                    test_feed_dict={ self.input: input_batch, target: target_batch }
                    self.sess.run(tf.assign_add(test_loss_tmp, sample_loss), feed_dict=test_feed_dict)

                loss  = self.sess.run(tf.assign(self.test_loss, tf.div(test_loss_tmp, tf_iters)))
                t_log = self.sess.run(self.test_loss_summary)

                print('[Test]: loss = %.4f'%loss)

                self.summary_writer.add_summary(t_log, step)
                self.summary_writer.flush()

    def draw_samples(inputs, samples, targets, step):
        # create the output dir if it doesn't exist
        output_dir = os.path.join(self.output_dir, 'samples')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        nr_samples = inputs.shape[0]
        for i in xrange(nr_samples):
            L = (inputs[i,:,:,:] + 128).astype(np.int16)
            sA = ((sample[i,:,:,0] + 1) * 128).astype(np.int16)
            sB = ((sample[i,:,:,1] + 1) * 128).astype(np.int16)
            gA = ((targets[i,:,:,0] + 1) * 128).astype(np.int16)
            gB = ((targets[i,:,:,1] + 1) * 128).astype(np.int16)
            sample_im = cv2.merge((L, sA, sB))
            gt_im     = cv2.merge((L, gA, gB))
            sample_im = cv2.cvtColor(sample_im, cv2.COLOR_LAB2BGR)
            gt_im     = cv2.cvtColor(gt_im, cv2.COLOR_LAB2BGR)

            im = np.zeros((self.train_shape[0], 2 * self.train_shape[1], 3), dtype=np.int16)
            im[:,:self.train_shape[1],:] = gt_im
            im[:,self.train_shape[1]:,:] = sample_im

            im_name = '%08d_%04d.jpg'%(step, i)
            output_path = os.path.join(output_dir, im_name)
            cv2.imwrite(output_path, im)


if __name__ == '__main__':
    facedb = imdb('face_images')
    #facedb.draw_roidb('output')

    with tf.Session() as sess:
        handler = colorization(sess, facedb, 4, [128, 128], 'output/')
        handler.train(mode='mean')
