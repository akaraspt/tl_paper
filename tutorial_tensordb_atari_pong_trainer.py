#! /usr/bin/python
# -*- coding: utf8 -*-
"""
To understand Reinforcement Learning, we let computer to learn how to play
Pong game from the original screen inputs. Before we start, we highly recommend
you to go through a famous blog called “Deep Reinforcement Learning: Pong from
Pixels” which is a minimalistic implementation of deep reinforcement learning by
using python-numpy and OpenAI gym environment.

The code here is the reimplementation of Karpathy's Blog by using TensorLayer.

Link
-----
http://karpathy.github.io/2016/05/31/rl/
"""
import tensorflow as tf
import tensorlayer as tl
import gym
import numpy as np
import time
# hyperparameters
image_size = 80
D = image_size * image_size
H = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
# render = False      # display the game environment
# resume = False      # load existing policy network
# model_file_name = "model_pong"
np.set_printoptions(threshold=np.nan)

from tensorlayer.db import TensorDB
# db = TensorDB(ip='localhost', port=27017, db_name='atari', user_name=None, password=None) #<- if none password
# db = TensorDB(ip='146.169.33.34', port=27020, db_name='DRL', user_name='tensorlayer', password='Tensorlayer123', studyID='1')
db = TensorDB(ip='146.169.15.140', port=27017, db_name='DRL', user_name=None, password=None, studyID='1')

# def prepro(I):
#     """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
#     I = I[35:195]
#     I = I[::2,::2,0]
#     I[I == 144] = 0
#     I[I == 109] = 0
#     I[I != 0] = 1
#     return I.astype(np.float).ravel()

# env = gym.make("Pong-v0")
# observation = env.reset()
# prev_x = None
# running_reward = None
# reward_sum = 0
# episode_number = 0

# xs, ys, rs = [], [], []
# observation for training and inference
states_batch_pl = tf.placeholder(tf.float32, shape=[None, D])
# policy network
net = tl.layers.InputLayer(states_batch_pl, name='input')
net = tl.layers.DenseLayer(net, n_units=H, act=tf.nn.relu, name='relu1')
net = tl.layers.DenseLayer(net, n_units=3, act=tf.identity, name='output')
probs = net.outputs
sampling_prob = tf.nn.softmax(probs)

actions_batch_pl = tf.placeholder(tf.int32, shape=[None])
discount_rewards_batch_pl = tf.placeholder(tf.float32, shape=[None])
loss = tl.rein.cross_entropy_reward_loss(probs, actions_batch_pl,
                                                    discount_rewards_batch_pl)
train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)

with tf.Session() as sess:
    tl.layers.initialize_global_variables(sess)
    # if resume:
    #     load_params = tl.files.load_npz(name=model_file_name+'.npz')
    #     tl.files.assign_params(sess, load_params, net)
    net.print_params()
    net.print_layers()

    start_time = time.time()
    game_number = 0
    n = 0
    total_n_examples = 0
    while True:
        is_found = False
        while is_found is False:
            ## read on
            data, f_id = db.find_one_params(args={'type': 'train_data'})
            if (data is not False):
                epx, epy, epr = data
                db.del_params(args={'type': 'train_data', 'f_id': f_id})
                is_found = True
            else:
                # print("Waiting training data")
                time.sleep(0.5)
            ## read all
            # temp = db.find_all_params(args={'type': 'train'})
            # if (temp is not False):
            #     epx = temp[0][0]
            #     for i in range(1, len(temp[0])):
            #         epx = np.append(epx, temp[i][0], axis = 0)
            #     epy = temp[0][1]
            #     for i in range(1, len(temp[1])):
            #         epy = np.append(epy, temp[i][1], axis = 0)
            #     epr = temp[0][2]
            #     for i in range(1, len(temp[2])):
            #         epr = np.append(epr, temp[i][2], axis = 0)
            #     is_found = True
            #     break
        disR = tl.rein.discount_episode_rewards(epr, gamma)
        disR -= np.mean(disR)
        disR /= np.std(disR)

        sess.run(train_op,{
                states_batch_pl: epx,
                actions_batch_pl: epy,
                discount_rewards_batch_pl: disR
            })
        n_examples = epx.shape[0]
        total_n_examples += n_examples
        print("[*] Update {}: n_examples: {} / total averaged speed: {} examples/second".format(n, n_examples,
                                round(total_n_examples/(time.time() - start_time), 2)))
        n += 1

        if n % 10 == 0:
            db.del_params(args={'type': 'network_parameters'})
            db.save_params(sess.run(net.all_params), args={'type': 'network_parameters'})#, file_name='network_parameters')
