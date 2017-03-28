#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.db import TensorDB
from tensorlayer.layers import set_keep
import time
import shutil

"""Examples of Stacked Denoising Autoencoder, Dropout, Dropconnect and CNN.
This tutorial uses placeholder to control all keeping probabilities,
so we need to set the non-one probabilities during training, and set them to 1
during evaluating and testing.
$ Set keeping probabilities.
>>> feed_dict = {x: X_train_a, y_: y_train_a}
>>> feed_dict.update( network.all_drop )
$ Set all keeping probabilities to 1 for evaluating and testing.
>>> dp_dict = tl.utils.dict_to_one( network.all_drop )
>>> feed_dict = {x: X_train_a, y_: y_train_a}
>>> feed_dict.update(dp_dict)
Alternatively, if you don't want to use placeholder to control them, you can
build different inferences for training, evaluating and testing,
and all inferences share the same model parameters.
(see tutorial_ptb_lstm.py)
"""


def main_test_layers():
    X_train, y_train, X_val, y_val, X_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1, 784))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    sess = tf.InteractiveSession()

    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    # Note: the softmax is implemented internally in tl.cost.cross_entropy(y, y_)
    # to speed up computation, so we use identity in the last layer.
    # see tf.nn.sparse_softmax_cross_entropy_with_logits()
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=800,
                                   act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=800,
                                   act=tf.nn.relu, name='relu2')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    network = tl.layers.DenseLayer(network, n_units=10,
                                   act=tf.identity,
                                   name='output_layer')

    y = network.outputs
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    # cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    cost = tl.cost.cross_entropy(y, y_, name='cost')

    # You can add more penalty to the cost function as follow.
    # cost = cost + tl.cost.maxnorm_regularizer(1.0)(network.all_params[0]) + tl.cost.maxnorm_regularizer(1.0)(network.all_params[2])
    # cost = cost + tl.cost.lo_regularizer(0.0001)(network.all_params[0]) + tl.cost.lo_regularizer(0.0001)(network.all_params[2])
    # cost = cost + tl.cost.maxnorm_o_regularizer(0.001)(network.all_params[0]) + tl.cost.maxnorm_o_regularizer(0.001)(network.all_params[2])

    params = network.all_params
    # train
    n_epoch = 100
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 5
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost)

    tl.layers.initialize_global_variables(sess)

    network.print_params()
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train,
                                                           batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(network.all_drop)  # enable dropout or dropconnect layers
            sess.run(train_op, feed_dict=feed_dict)

            # The optional feed_dict argument allows the caller to override the value of tensors in the graph. Each key in feed_dict can be one of the following types:
            # If the key is a Tensor, the value may be a Python scalar, string, list, or numpy ndarray that can be converted to the same dtype as that tensor. Additionally, if the key is a placeholder, the shape of the value will be checked for compatibility with the placeholder.
            # If the key is a SparseTensor, the value should be a SparseTensorValue.

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable all dropout/dropconnect/denoising layers
            feed_dict = {x: X_train, y_: y_train}
            feed_dict.update(dp_dict)
            print("   train loss: %f" % sess.run(cost, feed_dict=feed_dict))
            dp_dict = tl.utils.dict_to_one(network.all_drop)
            feed_dict = {x: X_val, y_: y_val}
            feed_dict.update(dp_dict)
            print("   val loss: %f" % sess.run(cost, feed_dict=feed_dict))
            print("   val acc: %f" % np.mean(y_val ==
                                             sess.run(y_op, feed_dict=feed_dict)))
            try:
                # You can visualize the weight of 1st hidden layer as follow.
                tl.visualize.W(network.all_params[0].eval(), second=10,
                               saveable=True, shape=[28, 28],
                               name='w1_' + str(epoch + 1), fig_idx=2012)
                # You can also save the weight of 1st hidden layer to .npz file.
                # tl.files.save_npz([network.all_params[0]] , name='w1'+str(epoch+1)+'.npz')
            except:
                raise Exception("You should change visualize_W(), if you want \
                            to save the feature images for different dataset")

    print('Evaluation')
    dp_dict = tl.utils.dict_to_one(network.all_drop)
    feed_dict = {x: X_test, y_: y_test}
    feed_dict.update(dp_dict)
    print("   test loss: %f" % sess.run(cost, feed_dict=feed_dict))
    print("   test acc: %f" % np.mean(y_test == sess.run(y_op,
                                                         feed_dict=feed_dict)))

    # Add ops to save and restore all the variables, including variables for training.
    # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)

    # You can also save the parameters into .npz file.
    tl.files.save_npz(network.all_params, name='model.npz')
    # You can only save one parameter as follow.
    # tl.files.save_npz([network.all_params[0]] , name='model.npz')
    # Then, restore the parameters as follow.
    # load_params = tl.files.load_npz(path='', name='model.npz')
    # tl.files.assign_params(sess, load_params, network)

    # In the end, close TensorFlow session.
    sess.close()


def main_test_cnn_layer():
    """Reimplementation of the TensorFlow official MNIST CNN tutorials:
    - https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py
    More TensorFlow official CNN tutorials can be found here:
    - tutorial_cifar10.py
    - https://www.tensorflow.org/versions/master/tutorials/deep_cnn/index.html
    - For simplified CNN layer see "Convolutional layer (Simplified)"
      in read the docs website.
    """

    sess = tf.InteractiveSession()

    # Define the batchsize at the begin, you can give the batchsize in x and y_
    # rather than 'None', this can allow TensorFlow to apply some optimizations
    # – especially for convolutional layers.
    batch_size = 128

    x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])  # [batch_size, height, width, channels]
    y_ = tf.placeholder(tf.int64, shape=[batch_size, ])

    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv2d(network, n_filter=32, filter_size=(5, 5), strides=(1, 1),
                               act=tf.nn.relu, padding='SAME', name='cnn1')
    network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                  padding='SAME', name='pool_layer1')
    network = tl.layers.Conv2d(network, n_filter=64, filter_size=(5, 5), strides=(1, 1),
                               act=tf.nn.relu, padding='SAME', name='cnn2')
    network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                  padding='SAME', name='pool_layer2')
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=256,
                                   act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=10,
                                   act=tf.identity,
                                   name='output')

    y = network.outputs

    cost = tl.cost.cross_entropy(y, y_, 'cost')

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    n_epoch = 200
    learning_rate = 0.0001
    print_freq = 10

    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    tl.layers.initialize_global_variables(sess)
    network.print_params()
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(
                X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(network.all_drop)  # enable noise layers
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                    X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err;
                train_acc += ac;
                n_batch += 1
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(
                    X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err;
                val_acc += ac;
                n_batch += 1
            print("   val loss: %f" % (val_loss / n_batch))
            print("   val acc: %f" % (val_acc / n_batch))
            try:
                tl.visualize.CNN2d(network.all_params[0].eval(),
                                   second=10, saveable=True,
                                   name='cnn1_' + str(epoch + 1), fig_idx=2012)
            except:
                raise Exception(
                    "# You should change visualize.CNN(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(
            X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err;
        test_acc += ac;
        n_batch += 1
    print("   test loss: %f" % (test_loss / n_batch))
    print("   test acc: %f" % (test_acc / n_batch))


### Master node ###


def create_mnist_dataset(db):
    data, f_id = db.find_one_params(args={'type': 'mnist_dataset'})
    # If cannot find MNIST dataset in TensorDB
    if not data:
        # Download and upload MNIST dataset to TensorDB
        X_train, y_train, X_val, y_val, X_test, y_test = \
            tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
        f_id = db.save_params(
            [X_train, y_train, X_val, y_val, X_test, y_test],
            args={'type': 'mnist_dataset'}
        )
        shutil.rmtree('./data/mnist')


def create_jobs(db, job_name, models_dict):
    # job = db.find_one_job(args={'job_name': job_name})
    # if not job:
    #     job_idx = 1
    #     for model, params_dict in models_dict.iteritems():
    #         n_jobs = len(params_dict.itervalues().next())
    #         for j in range(n_jobs):
    #             job_dict = {'model': model, 'job_name': job_name, 'job_id': job_idx}
    #             for k, v in params_dict.iteritems():
    #                 job_dict.update({k: v[j]})
    #             db.save_job(args=job_dict)
    #             job_idx += 1
    # else:
    #     print("You have already submitted this job.")
    for model, params_dict in models_dict.iteritems():
        n_jobs = len(params_dict.itervalues().next())
        for j in range(n_jobs):
            job_dict = {'model': model}
            for k, v in params_dict.iteritems():
                job_dict.update({k: v[j]})
            db.save_job(args=job_dict)


def start_workers(db):
    job_ids = []
    for job in db.get_all_jobs():
        job_ids.append(str(job['_id']))

    # Check how many available workers
    workers = ['node01', 'node02', 'node03']

    def submit_job(node_name, job_id):
        print('Assign job: {} to {}'.format(job_id, node_name))
        worker(job_id)

    # Submit jobs to all workers
    submit_job(workers[0], job_ids[0])
    submit_job(workers[1], job_ids[1])
    submit_job(workers[2], job_ids[2])


def master():
    db = TensorDB(ip='146.169.33.34', port=27020, db_name='TransferGan', user_name='akara', password='DSIGPUfour')
    create_mnist_dataset(db=db)
    create_jobs(db=db, job_name="cv_mnist", models_dict={
        "cnn": {
            "learning_rate": [0.01, 0.001, 0.001],
            "n_layers": [3, 5, 7],
            "n_filters": [64, 128, 256]
        },
        "mlp": {
            "learning_rate": [0.05, 0.005],
            "n_layers": [4, 6],
        }
    })
    start_workers(db=db)


### Workder node ###


def load_mnist_data(db):
    data, f_id = db.find_one_params(args={'type': 'mnist_dataset'})
    if not data:
        raise Exception("MNIST dataset not found !!")
    X_train, y_train, X_val, y_val, X_test, y_test = data

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    return X_train, y_train, X_val, y_val, X_test, y_test


def worker(job_id):
    db = TensorDB(ip='146.169.33.34', port=27020, db_name='TransferGan', user_name='akara', password='DSIGPUfour')
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data(db=db)

    from bson.objectid import ObjectId
    print(db.find_one_job(args={'_id': ObjectId(job_id)}))


### Main ###


def main():
    master()


if __name__ == '__main__':
    main()
