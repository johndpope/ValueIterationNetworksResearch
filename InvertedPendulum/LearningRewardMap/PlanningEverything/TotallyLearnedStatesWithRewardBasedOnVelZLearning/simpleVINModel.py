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
    ch_i = config.ch_i
    batch_size = tf.shape(state_input)[0];

    if (weights == None):
        #Reward Map
        cost_hidden1 = tf.Variable(np.random.randn(1, config.vel_hidden1) * 0.001, dtype=tf.float32)
        cost_bias1 = tf.Variable(np.random.randn(1, config.vel_hidden1) * 0.001, dtype=tf.float32)
        cost_output = tf.Variable(np.random.randn(config.vel_hidden1, numstates) * 0.001, dtype=tf.float32)
        cost_biasOut = tf.Variable(np.random.randn(1, numstates) * 0.001, dtype=tf.float32)

        # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
        w_fb  = tf.Variable(np.random.randn(width, 1, 1) * 0.001, dtype=tf.float32)

        #State weights
        state_hidden1 = tf.Variable(np.random.randn(state_dim - 1, config.state_hidden1) * 0.001, dtype=tf.float32)
        state_bias1 = tf.Variable(np.random.randn(1, config.state_hidden1) * 0.001, dtype=tf.float32)
        state_output = tf.Variable(np.random.randn(config.state_hidden1,1) * 0.001, dtype=tf.float32)
        state_biasOut = tf.Variable(np.random.randn(1, 1) * 0.001, dtype=tf.float32)

        #output weights
        w_h1   = tf.Variable(np.random.randn(width + state_dim, config.hidden1) * 0.001, dtype=tf.float32)
        bias1 = tf.Variable(np.random.randn(1, config.hidden1) * 0.001, dtype=tf.float32)
        bias_o = tf.Variable(np.random.randn(1, numactions) * 0.001, dtype=tf.float32)
        w_o = tf.Variable(np.random.randn(config.hidden1, numactions) * 0.001, dtype=tf.float32)

    else:
        cost_hidden1 = weights[0];
        cost_bias1 = weights[1];
        cost_output = weights[2];
        cost_biasOut = weights[3];

        w_fb  = weights[4]

        state_hidden1 = weights[5]
        state_bias1 = weights[6]
        state_output = weights[7]
        state_biasOut = weights[8]

        w_h1   = weights[9]
        bias1 = weights[10]
        w_o = weights[11]
        bias_o = weights[12]

    #Make Reward Maps
    state_input_Transpose = tf.transpose(state_input, perm=[1,0])
    velocities = tf.reshape(state_input_Transpose[2],[-1, 1])
    c_hidden1 = tf.nn.relu(tf.matmul(velocities, cost_hidden1) + cost_bias1)
    c = tf.nn.tanh(tf.matmul(c_hidden1, cost_output) + cost_biasOut) + 1
    c = tf.reshape(c, [batch_size, numstates, 1])

    v = tf.multiply(c, circularConv(c, w_fb))
    #value iterations
    for i in range(0, k-1):
        v = tf.multiply(c, circularConv(v, w_fb))

    filt = tf.reshape(tf.constant(np.identity(width), tf.float32), [width, 1, width])
    q = circularConv(v, filt)

    # Calculate Position
    state_h1  = tf.nn.relu(tf.matmul(tf.transpose(state_input_Transpose[0: state_dim - 1]), state_hidden1) + state_bias1)
    state = tf.nn.tanh(tf.matmul(state_h1, state_output) + state_biasOut) + 1
    S1 = tf.reshape(tf.cast(tf.floordiv(state, 2.0/numstates), dtype=tf.int32), [-1])


    #Select the conv-net channels at the state position
    ins1 = tf.range(batch_size)
    idx_in = tf.transpose(tf.stack([ins1, S1]), [1,0])

    #Output action
    q_out = tf.gather_nd(q, idx_in, name="q_out")
    inputs = tf.concat([state_input, q_out], axis=1)
    hiddenLayer1 = tf.nn.relu(tf.matmul(inputs, w_h1) + bias1)
    output = tf.nn.tanh(tf.matmul(hiddenLayer1, w_o) + bias_o);
    return state_input, output, [cost_hidden1, cost_bias1, cost_output,
        cost_biasOut, w_fb, state_hidden1, state_bias1, state_output,
        state_biasOut, w_h1, bias1, w_o, bias_o]
