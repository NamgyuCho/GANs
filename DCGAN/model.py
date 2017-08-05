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
        :param gf_dim: (optional) Dimension
        :param df_dim:
        :param gfc_dim:
        :param dfc_dim:
        :param c_dim:
        :param dataset_name:
        :param input_fname_pattern:
        :param checkpoint_dir:
        :param sample_dir:
        """