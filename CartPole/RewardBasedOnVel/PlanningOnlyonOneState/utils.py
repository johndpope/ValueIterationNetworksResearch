import tensorflow as tf
import numpy as np

def circularConv(X, k):
    length = tf.shape(X)[1]
    filterLen = tf.shape(k)[0]
    Y = tf.concat([X[:, length - tf.floordiv(filterLen - 1, 2) :, :],X,X[:,: tf.floordiv(filterLen - 1, 2),:]],1)
    return tf.nn.conv1d(Y, k, stride=1, padding='VALID')

def specialConv(X, k):
    length = tf.shape(X)[1]
    filterLen = tf.shape(k)[0]
    Y = tf.concat([X[:, length - tf.floordiv(filterLen - 1, 2) :, :, :],X,X[:,: tf.floordiv(filterLen - 1, 2),:,:]],1)
    out = tf.nn.conv2d(Y, k, strides=(1,1,1,1), padding='SAME')
    return out[:,tf.floordiv(filterLen - 1, 2) : length + tf.floordiv(filterLen - 1, 2) , :, :]

def conv2d(X, k):
    return tf.nn.conv2d(X, k, strides=(1,1,1,1), padding='SAME')

def conv1d(X, k):
    return tf.nn.conv1d(X,k, stride=1, padding='SAME')

def theta(x, y, epsilon=1.0e-12):
        angle = tf.where(tf.greater_equal(y, 0.0), tf.acos(x), 2 * np.pi - tf.acos(x) - epsilon)
        return angle
