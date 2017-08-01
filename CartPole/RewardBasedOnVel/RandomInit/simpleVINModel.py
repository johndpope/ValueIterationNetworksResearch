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
        reward_hidden1 = tf.Variable(np.random.randn(3, config.vel_hidden1) * 0.001, dtype=tf.float32)
        reward_bias1 = tf.Variable(np.random.randn(1, config.vel_hidden1) * 0.001, dtype=tf.float32)
        reward_output = tf.Variable(np.random.randn(numstates, numstates, config.vel_hidden1) * 0.001, dtype=tf.float32)
        reward_biasOut = tf.Variable(np.random.randn(1,numstates, numstates) * 0.001, dtype=tf.float32)

        w = tf.Variable(np.random.randn(width, width, 1, ch_q) * 0.001, dtype=tf.float32)
        # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
        w_fb  = tf.Variable(np.random.randn(width, width, 1, ch_q) * 0.001, dtype=tf.float32)

        #output weights
        w_h1   = tf.Variable(np.random.randn(ch_q + state_dim, config.hidden1) * 0.001, dtype=tf.float32)
        bias1 = tf.Variable(np.random.randn(1, config.hidden1) * 0.001, dtype=tf.float32)
        w_o = tf.Variable(np.random.randn(config.hidden1, numactions) * 0.001, dtype=tf.float32)
        bias_o = tf.Variable(np.random.randn(1, numactions) * 0.001, dtype=tf.float32)

    else:
        reward_hidden1 = weights[0];
        reward_bias1 = weights[1];
        reward_output = weights[2];
        reward_biasOut = weights[3];
        w     = weights[4]
        w_fb  = weights[5]
        w_h1   = weights[6]
        bias1 = weights[7]
        w_o = weights[8]
        bias_o = weights[9]

    #Make Reward Maps
    state_input_Transpose = tf.transpose(state_input, perm=[1,0])
    velocities = tf.stack([state_input_Transpose[2], state_input_Transpose[3], state_input_Transpose[4]])
    velocities = tf.transpose(velocities, perm=[1,0])
    r_hidden1 = tf.nn.relu(tf.matmul(velocities, reward_hidden1) + reward_bias1)
    r = tf.nn.tanh(tf.tensordot(r_hidden1, reward_output, axes=[[1],[2]]) + reward_biasOut)
    r = tf.reshape(r, [batch_size, numstates, numstates, 1])

    q = conv2d(r, w)
    v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")
    wwfb = tf.concat([w, w_fb], 2)

    #value iterations
    for i in range(0, k-1):
        rv = tf.concat([r, v], 3)
        q = conv2d(rv, wwfb)
        v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")
    q = conv2d(tf.concat([r, v], 3), wwfb)

    # Calculate Position

    S1 = tf.cast(tf.floordiv(state_input_Transpose[0] + 1.1, 2.2/(numstates)), tf.int32)
    S1 = tf.minimum(tf.maximum(S1, 0),numstates - 1)
    S2 = tf.cast(tf.floordiv(state_input_Transpose[1] + .2, (.4)/(numstates)), tf.int32)

    #Select the conv-net channels at the state position
    ins1 = tf.range(batch_size)
    idx_in = tf.transpose(tf.stack([ins1, S1, S2]), [1,0])
    #idx_in = tf.Print(idx_in, [idx_in], "INDEX IN:")

    #Output action
    q_out = tf.gather_nd(q, idx_in, name="q_out")
    inputs = tf.concat([state_input, q_out], axis=1)
    hiddenLayer1 = tf.nn.relu(tf.matmul(inputs, w_h1) + bias1)
    output = tf.nn.tanh(tf.matmul(hiddenLayer1, w_o) + bias_o);
    return state_input, output, [reward_hidden1, reward_bias1, reward_output,
        reward_biasOut, w, w_fb, w_h1, bias1, w_o, bias_o]
