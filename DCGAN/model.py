from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
        """

        :param sess: TensorFlow session.
        :param batch_size: The size of batch. Should be specified before training.
        :param y_dim: (optional) Dimension of y [None].
        :param z_dim: (optional) Dimension of z [100].
        :param gf_dim: (optional) Dimension of gen filters in first conv layer [64].
        :param df_dim: (optional) Dimension of discrim filters in first conv layer [64].
        :param gfc_dim: (optional) Dimension of gen units for fully connected layer [1024].
        :param dfc_dim: (optional) Dimension of discrim units for fully connected layer [1024].
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1 [3].
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization: deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        if self.dataset_name == 'mnist':
            self.data_X, self.data_Y = self.load_mnist()
            self.c_dim = self.data_X[0].shape[-1]
        else:
            self.data = glob(os.path.join('./data', self.dataset_name, self.input_fname_pattern))
            imreadImg = imread(self.data[0])
            if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1

        self.grayscale = (self.c_dim == 1)
        self.build_model()

    def build_mode(self):
        if self.y_dim:
            self.y_dim = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

