import tensorflow as tf
import numpy as np
import math
from utils import *

def VIN(state_input, state_dim, action_dim, config, weights=None):
    numactions = action_dim;
    numstates = config.numstates;
    k = config.k
    width = config.width
    assert width % 2 == 1
    assert config.numstates % 2 == 0
    ch_i = config.ch_i
    ch_h = config.ch_h
    ch_q = config.ch_q
    hiddenUnits = config.hidden1
    state_batch_size = tf.shape(state_input)[0];

    if (weights == None):
        #Weights for each action's reward
        w     = tf.Variable(np.random.randn(width, 1, ch_q) * 0.001, dtype=tf.float32)

        # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
        w_fb  = tf.Variable(np.random.randn(width, 1, ch_q) * 0.001, dtype=tf.float32)

        #Output weights
        bias1 = tf.Variable(np.random.randn(1, hiddenUnits) * 0.001, dtype=tf.float32)
        w_h1   = tf.Variable(np.random.randn(ch_q + state_dim, hiddenUnits) * 0.001, dtype=tf.float32)
        bias2 = tf.Variable(np.random.randn(1, numactions) * 0.001, dtype=tf.float32)
        w_o = tf.Variable(np.random.randn(hiddenUnits, numactions) * 0.001, dtype=tf.float32)

        #Intialization of half of the reward map
        initial = tf.scalar_mul(2 * np.pi / config.numstates, tf.range(0, config.numstates/2))
        intial = -1 * tf.square(initial)
        initial = tf.cast(intial, tf.float32)
        initial = tf.reshape(intial, [config.batchsize, int(config.numstates/2), config.ch_i])
        X = tf.Variable(initial * 0.001, dtype=tf.float32)
    else:
        w     = weights[0]
        w_fb  = weights[1]
        bias1 = weights[2]
        w_h1   = weights[3]
        bias2 = weights[4]
        w_o = weights[5]
        X = weights[6]

    #Making the symmetric reward map
    r = tf.concat([X, tf.reverse(X, [1])], axis = 1)
    q = circularConv(r, w)
    v = tf.reduce_max(q, axis=2, keep_dims=True, name="v")
    wwfb = tf.concat([w, w_fb], 1)

    #Value Iteration
    for i in range(0, k-1):
        rv = tf.concat([r, v], 2)
        q = circularConv(rv, wwfb)
        v = tf.reduce_max(q, axis=2, keep_dims=True, name="v")

    # do one last convolution
    q = circularConv(tf.concat([r, v], 2), wwfb)

    # Select the conv-net channels at the state position
    position = tf.transpose(state_input, perm=[1,0])
    angle = theta(position[0], position[1])
    S1 = tf.cast(tf.floordiv(angle, 2*math.pi / numstates), tf.int32)
    ins1 = tf.zeros(tf.shape(S1), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, S1]), [1,0])

    #Concat action values to observations
    q_out = tf.gather_nd(q, idx_in, name="q_out")
    inputs = tf.concat([state_input, q_out], axis=1)

    #Output action
    hiddenLayer1 = tf.nn.relu(tf.matmul(inputs, w_h1) + bias1)
    output = tf.nn.tanh(tf.matmul(hiddenLayer1, w_o) + bias2, name="output")
    return state_input, output, [w, w_fb, bias1, w_h1, bias2, w_o, X]
