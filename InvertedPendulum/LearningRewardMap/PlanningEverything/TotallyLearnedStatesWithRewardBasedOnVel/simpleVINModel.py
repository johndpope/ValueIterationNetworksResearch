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
    ch_q = config.ch_q
    batch_size = tf.shape(state_input)[0];

    if (weights == None):
        #Reward Map
        reward_hidden1 = tf.Variable(np.random.randn(1, config.vel_hidden1) * 0.001, dtype=tf.float32)
        reward_bias1 = tf.Variable(np.random.randn(1, config.vel_hidden1) * 0.001, dtype=tf.float32)
        reward_output = tf.Variable(np.random.randn(config.vel_hidden1, numstates) * 0.001, dtype=tf.float32)
        reward_biasOut = tf.Variable(np.random.randn(1, numstates) * 0.001, dtype=tf.float32)

        w = tf.Variable(np.random.randn(width, 1, ch_q) * 0.001, dtype=tf.float32)
        # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
        w_fb  = tf.Variable(np.random.randn(width, 1, ch_q) * 0.001, dtype=tf.float32)

        #State weights
        state_hidden1 = tf.Variable(np.random.randn(state_dim - 1, config.state_hidden1) * 0.001, dtype=tf.float32)
        state_bias1 = tf.Variable(np.random.randn(1, config.state_hidden1) * 0.001, dtype=tf.float32)
        state_output = tf.Variable(np.random.randn(config.state_hidden1,1) * 0.001, dtype=tf.float32)
        state_biasOut = tf.Variable(np.random.randn(1, 1) *0.001, dtype=tf.float32)

        #output weights
        w_h1   = tf.Variable(np.random.randn(ch_q + state_dim, config.hidden1) * 0.001, dtype=tf.float32)
        bias1 = tf.Variable(np.random.randn(1, config.hidden1) * 0.001, dtype=tf.float32)
        w_h2 = tf.Variable(np.random.randn(config.hidden1, config.hidden2) * 0.001, dtype=tf.float32)
        bias2 = tf.Variable(np.random.randn(1, config.hidden2) * 0.001, dtype=tf.float32)
        bias_o = tf.Variable(np.random.randn(1, numactions) * 0.001, dtype=tf.float32)
        w_o = tf.Variable(np.random.randn(config.hidden2, numactions) * 0.001, dtype=tf.float32)

    else:
        reward_hidden1 = weights[0];
        reward_bias1 = weights[1];
        reward_output = weights[2];
        reward_biasOut = weights[3];

        w     = weights[4]
        w_fb  = weights[5]

        state_hidden1 = weights[6]
        state_bias1 = weights[7]
        state_output = weights[8]
        state_biasOut = weights[9]

        w_h1   = weights[10]
        bias1 = weights[11]
        w_h2 = weights[12]
        bias2 = weights[13]
        w_o = weights[14]
        bias_o = weights[15]

    #Make Reward Maps
    state_input_Transpose = tf.transpose(state_input, perm=[1,0])
    velocities = tf.reshape(state_input_Transpose[2],[-1, 1])
    r_hidden1 = tf.nn.relu(tf.matmul(velocities, reward_hidden1) + reward_bias1)
    r = tf.nn.tanh(tf.matmul(r_hidden1, reward_output) + reward_biasOut)
    r = tf.reshape(r, [batch_size, numstates, 1])

    q = circularConv(r, w)
    v = tf.reduce_max(q, axis=2, keep_dims=True, name="v")
    wwfb = tf.concat([w, w_fb], 1)

    #value iterations
    for i in range(0, k-1):
        rv = tf.concat([r, v], 2)
        q = circularConv(rv, wwfb)
        v = tf.reduce_max(q, axis=2, keep_dims=True, name="v")
    q = circularConv(tf.concat([r, v], 2), wwfb)

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
    hiddenLayer2 = tf.nn.relu(tf.matmul(hiddenLayer1, w_h2) + bias2)
    output = tf.nn.tanh(tf.matmul(hiddenLayer1, w_o) + bias_o);
    return state_input, output, [reward_hidden1, reward_bias1, reward_output,
        reward_biasOut, w, w_fb, state_hidden1, state_bias1, state_output,
        state_biasOut, w_h1, bias1, w_h2, bias2, w_o, bias_o]
