import tensorflow as tf
import numpy as np

tf.apps.flags.DEFINE_integer('numactions', 40, 'Number of Actions')
tf.apps.flags.DEFINE_integer('numstates', 100, 'Number of States')
tf.apps.flags.DEFINE_integer('k', 30, 'Number of Value Iterations')
tf.app.flags.DEFINE_integer('width', 3, 'Size of conv filter (~state connectivity)')
tf.app.flags.DEFINE_integer('ch_i', 2, 'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h', 150, 'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q', 10, 'Channels in q layer (~actions)')
tf.apps.flag.DEFINE_integer('batchsize', 12, 'Batch Size')
