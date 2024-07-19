import os
import math
import logging

import numpy as np


import tensorflow as tf

@tf.function
def sigmoid(x):
    return tf.math.sigmoid(x)

@tf.function
def get_state(data, t, n):
    t = tf.cast(t, tf.int32)
    n = tf.cast(n, tf.int32)
    d = t - n + 1
    
    block = tf.cond(
        tf.greater_equal(d, 0),
        lambda: data[d:t + 1],
        lambda: tf.concat([tf.repeat(data[0], tf.abs(d)), data[:t + 1]], axis=0)
    )
    
    diffs = block[1:] - block[:-1]
    return sigmoid(diffs)  # This should return a tensor of shape (n-1,)