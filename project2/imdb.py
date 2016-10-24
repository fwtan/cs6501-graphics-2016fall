#!/usr/bin/env python

import os, sys
import cv2, copy
import math, cPickle
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt


class imdb(object):
    """Image database."""

    def __init__(self, name):
        self.name = name
        roidb = self.load_dataset(name)
        n_all = len(roidb)
        self.n_val = int(0.1 * n_all)
        self.val_roidb   = roidb[:self.n_val]
        self.train_roidb = roidb[self.n_val:]
        self.n_train  = len(self.train_roidb)

        self.append_flipped_images()
        self.append_cropped_images()
        self.append_scaled_images()

    def load_dataset(self, name, ext='.jpg'):
        fnames = [f for f in os.listdir(osp.join('data', name)) if osp.splitext(osp.basename(f))[-1] == ext]
        return [self.load_image(f) for f in fnames]

    def load_image(self, fname):
        return {'image'   : osp.join('data', self.name, fname),
                'flipped' : False,
                'scale'   : 1.0,
                'roi'     : [0.0, 0.0, 1.0, 1.0]}

    def append_flipped_images(self):
        for i in xrange(self.n_train):
            entry = copy.deepcopy(self.train_roidb[i])
            entry['flipped'] = True
            self.train_roidb.append(entry)

    def append_cropped_images(self):
        for i in xrange(self.n_train):
            entry = copy.deepcopy(self.train_roidb[i])
            cropped_scale = 0.5 + 0.5 * np.random.random()
            cropped_x = (1.0 - cropped_scale) * np.random.random()
            cropped_y = (1.0 - cropped_scale) * np.random.random()
            entry['roi'] = [cropped_x, cropped_y, cropped_x + cropped_scale, cropped_y + cropped_scale]
            self.train_roidb.append(entry)

    def append_scaled_images(self):
        for i in xrange(self.n_train):
            entry = copy.deepcopy(self.train_roidb[i])
            entry['scale'] = 0.6 + 0.4 * np.random.random()
            self.train_roidb.append(entry)

    def draw_roidb(self, output_dir):
        i = 0
        for roi in self.train_roidb:
            img = cv2.imread(roi['image'], cv2.IMREAD_COLOR)
            h = float(img.shape[0])
            w = float(img.shape[1])
            cropped_roi = roi['roi']
            left   = int(np.minimum(np.maximum(0, w * cropped_roi[0]), w))
            right  = int(np.minimum(np.maximum(0, w * cropped_roi[2]), w))
            top    = int(np.minimum(np.maximum(0, h * cropped_roi[1]), h))
            bottom = int(np.minimum(np.maximum(0, h * cropped_roi[3]), h))
            img    = img[top:bottom, left:right, :]

            if roi['flipped']:
                img = cv2.flip(img, 1)

            img = (img.astype(np.float32) * roi['scale']).astype(np.int16)

            i += 1
            output_path = os.path.join(output_dir, '%05d_'%i + osp.basename(roi['image']))
            cv2.imwrite(output_path, img)


if __name__ == '__main__':
    facedb = imdb('face_images')
    facedb.draw_roidb('output')
