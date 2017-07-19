import tensorflow as tf
import numpy as np
import math
from simpleVINModel import *

# Hyper Parameters
tf.app.flags.DEFINE_integer('numstates',100, 'Number of States for VIN Planning')
tf.app.flags.DEFINE_integer('k', 30, 'Number of Value Iterations')
tf.app.flags.DEFINE_integer('width', 11, 'Size of conv filter (~state connectivity)')
tf.app.flags.DEFINE_integer('ch_i', 1, 'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h', 50, 'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q', 15, 'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('hidden1', 30, 'Size of final hidden layer')
tf.app.flags.DEFINE_integer('batchsize', 1, 'Batch Size')
#tf.app.flags.DEFINE_integer('state_batch_size', 12, 'State batch size (~num states per batch)')
#NOTE if changing conv type change in print out below

LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64

config = tf.app.flags.FLAGS

#Image input to VIN
if config.numstates % 2 == 0:
    X = tf.scalar_mul(-32.0 / config.numstates, tf.concat([tf.range(0, config.numstates/2), tf.range(config.numstates/2 - 1, -1, -1)], axis=0))
else:
    X = tf.scalar_mul(-32.0 / config.numstates, tf.concat([tf.range(0,config.numstates/2 + 1), tf.range(config.numstates/2, 0, -1)], axis=0))
X = tf.cast(X, tf.float32)
X = tf.reshape(X , [config.batchsize, config.numstates, config.ch_i])

#X = tf.ones([config.batchsize, config.numstates, config.ch_i])

class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self,sess,state_dim,action_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create actor network
        self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim)

        # create target actor network
        self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim,action_dim,self.net)

        # define training rules
        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        self.update_target()
        #self.load_network()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))

    def create_network(self,state_dim,action_dim):
        state_input = tf.placeholder("float", [None, state_dim])
        return VIN(X, state_input, state_dim, action_dim, config)

    def create_target_network(self,state_dim,action_dim,net):
        state_input = tf.placeholder("float",[None,state_dim])
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        _, action_output, _ = VIN(X, state_input, state_dim, action_dim, config, target_net)

        return state_input,action_output,target_update,target_net

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,q_gradient_batch,state_batch):
        self.sess.run(self.optimizer,feed_dict={
            self.q_gradient_input:q_gradient_batch,
            self.state_input:state_batch
            })

    def actions(self,state_batch):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:state_batch
            })

    def action(self,state):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:[state]
            })[0]


    def target_actions(self,state_batch):
        return self.sess.run(self.target_action_output,feed_dict={
            self.target_state_input:state_batch
            })

    # f fan-in size
    # def variable(self,shape,f):
    #     return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

    @staticmethod
    def get_settings():
        return "numstates; %d \n num value iterations; %d \n conv type; circular \n" \
            "conv width; %d \n channel_i; %d \n channel_h; %d \n channel_q; %d \n" \
            "hidden layer size; %d \n learning rate; %f \n TAU; %f \n batch size; %d \n reward;" \
            " \n New angle formula \n" % (config.numstates, config.k, config.width, config.ch_i,
            config.ch_h, config.ch_q, config.hidden1, LEARNING_RATE, TAU, BATCH_SIZE)
            #(np.array_str(X.eval())).replace("\n", " "))
'''
	def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
    def save_network(self,time_step):
        print 'save actor-network...',time_step
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''
