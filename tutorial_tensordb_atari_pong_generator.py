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
import time, os
import argparse
from bson.objectid import ObjectId

os.environ["CUDA_VISIBLE_DEVICES"]=""  # CPU

from tensorlayer.db import TensorDB
from tensorlayer.db import JobStatus

# This is to initialize the connection to your MondonDB server
# Note: make sure your MongoDB is reachable before changing this line
db = TensorDB(ip='IP_ADDRESS_OR_YOUR_MONGODB', port=27017, db_name='DATABASE_NAME', user_name=None, password=None, studyID='ANY_ID (e.g., mnist)')


def main(args):
    # hyperparameters
    image_size = 80
    D = image_size * image_size
    H = 200
    batch_size = 10
    # learning_rate = 1e-4
    gamma = 0.99
    # decay_rate = 0.99
    # render = False  # display the game environment
    # resume = False      # load existing policy network
    # model_file_name = "model_pong"
    np.set_printoptions(threshold=np.nan)

    def prepro(I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195]
        I = I[::2, ::2, 0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(np.float).ravel()

    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None
    running_reward = None
    reward_sum = 0
    episode_number = 0

    xs, ys, rs = [], [], []
    # observation for training and inference
    states_batch_pl = tf.placeholder(tf.float32, shape=[None, D])
    # policy network
    net = tl.layers.InputLayer(states_batch_pl, name='input')
    net = tl.layers.DenseLayer(net, n_units=H, act=tf.nn.relu, name='relu1')
    net = tl.layers.DenseLayer(net, n_units=3, act=tf.identity, name='output')
    probs = net.outputs
    sampling_prob = tf.nn.softmax(probs)

    # actions_batch_pl = tf.placeholder(tf.int32, shape=[None])
    # discount_rewards_batch_pl = tf.placeholder(tf.float32, shape=[None])
    # loss = tl.rein.cross_entropy_reward_loss(probs, actions_batch_pl,
    #                                                     discount_rewards_batch_pl)
    # train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)

    with tf.Session() as sess:
        tl.layers.initialize_global_variables(sess)
        # if resume:
        #     load_params = tl.files.load_npz(name=model_file_name+'.npz')
        #     tl.files.assign_params(sess, load_params, net)
        net.print_params()
        net.print_layers()

        start_time = time.time()
        game_number = 0
        while True:
            # if render: env.render()

            job = db.get_job(job_id=ObjectId(args.job_id))
            if job["status"] == JobStatus.TERMINATED:
                print("** Terminated by master node.")
                break

            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(D)
            x = x.reshape(1, D)
            prev_x = cur_x

            prob = sess.run(sampling_prob, feed_dict={states_batch_pl: x})
            # action. 1: STOP  2: UP  3: DOWN
            action = np.random.choice([1, 2, 3], p=prob.flatten())

            observation, reward, done, _ = env.step(action)
            reward_sum += reward
            xs.append(x)  # all observations in a episode
            ys.append(action - 1)  # all fake labels in a episode (action begins from 1, so minus 1)
            rs.append(reward)  # all rewards in a episode
            if done:
                episode_number += 1
                game_number = 0

                if episode_number % batch_size == 0:
                    # print('batch over...... updating parameters......')
                    print('batch over...... saving training data......')
                    epx = np.vstack(xs)
                    epy = np.asarray(ys)
                    epr = np.asarray(rs)
                    disR = tl.rein.discount_episode_rewards(epr, gamma)
                    disR -= np.mean(disR)
                    disR /= np.std(disR)

                    xs, ys, rs = [], [], []

                    print("[*] Generated {} examples".format(epx.shape[0]))

                    f_id = db.save_params([epx, epy, epr], args={'type': 'train_data'}, lz4_comp=True)  # , file_name='train_data')

                # if episode_number % (batch_size * 100) == 0:
                #     tl.files.save_npz(net.all_params, name=model_file_name+'.npz')

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
                reward_sum = 0
                observation = env.reset()  # reset env
                prev_x = None

            if reward != 0:
                print(('episode %d: game %d took %.5fs, reward: %f' %
                    (episode_number, game_number,
                        time.time() - start_time, reward)),
                    ('' if reward == -1 else ' !!!!!!!!'))
                start_time = time.time()

                if (episode_number % 20 == 0) and (game_number == 0):  ## Update model from Trainer
                    try:
                        params, f = db.find_one_params(args={'type': 'network_parameters'}, lz4_decomp=True)
                        if (params is not False):
                            tl.files.assign_params(sess, params, net)
                            print("[*] Update Model")
                    except:
                        continue

                game_number += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, required=True,
                        help="Job ID.")
    args = parser.parse_args()

    main(args)
