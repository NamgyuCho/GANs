"""
Implementing Deep Convolutional Generative Adversarial Network of simple version
(code: https://github.com/carpedm20/DCGAN-tensorflow/)
(http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)
"""

import os
import scipy.mist
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf
