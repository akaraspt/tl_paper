#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.db import TensorDB
from tensorlayer.layers import set_keep
import time
import shutil


def train_mlp(db, n_layers, lr, n_epochs):
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data(db=db, shape=(-1, 784))

    sess = tf.InteractiveSession()

    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    # MLP
    network = tl.layers.InputLayer(x, name='input_layer')
    for l in range(1, n_layers+1):
        network = tl.layers.DropoutLayer(network, keep=0.8, name='drop{}'.format(l))
        network = tl.layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu{}'.format(l))
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop'.format(n_layers+1))
    network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output_layer')
    y = network.outputs

    # Prediction
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    y_op = tf.to_int32(y_op)

    # Loss
    cost = tl.cost.cross_entropy(y, y_, name='cost')

    # Accuracy
    correct_prediction = tf.equal(y_op, y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train op
    batch_size = 128
    learning_rate = lr
    print_freq = 5

    params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost)

    tl.layers.initialize_global_variables(sess)

    network.print_params()
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epochs):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train,
                                                           batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epochs, time.time() - start_time))
            dp_dict = tl.utils.dict_to_one( network.all_drop )
            feed_dict = {x: X_train, y_: y_train}
            feed_dict.update(dp_dict)
            train_loss, train_acc = sess.run([cost, acc], feed_dict=feed_dict)
            print("   train loss: %f" % train_loss)
            print("   train acc: %f" % train_acc)
            dp_dict = tl.utils.dict_to_one( network.all_drop )
            feed_dict = {x: X_val, y_: y_val}
            feed_dict.update(dp_dict)
            valid_loss, valid_acc = sess.run([cost, acc], feed_dict=feed_dict)
            print("   val loss: %f" % valid_loss)
            print("   val acc: %f" % valid_acc)

    print('Evaluation')
    dp_dict = tl.utils.dict_to_one(network.all_drop)
    feed_dict = {x: X_test, y_: y_test}
    feed_dict.update(dp_dict)
    test_loss, test_acc = sess.run([cost, acc], feed_dict=feed_dict)
    print("   test loss: %f" % test_loss)
    print("   test acc: %f" % test_acc)

    # In the end, close TensorFlow session.
    sess.close()
    tl.layers.clear_layers_name()
    tf.reset_default_graph()


def train_cnn(db, n_cnn_layers, lr, n_epochs):
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data(db=db, shape=(-1, 28, 28, 1))

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y_ = tf.placeholder(tf.int32, shape=[None,])

    # CNN
    network = tl.layers.InputLayer(x, name='input_layer')

    if n_cnn_layers < 1 or n_cnn_layers > 2:
        raise Exception('Not yet support')
    filter_sizes = [32, 64]
    for l in range(n_cnn_layers):
        network = tl.layers.Conv2d(network, n_filter=filter_sizes[l], filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn{}'.format(l))
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool_layer{}'.format(l))
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=256, act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output')

    y = network.outputs

    # Prediction
    y_op = tf.argmax(y, 1)
    y_op = tf.to_int32(y_op)

    # Loss
    cost = tl.cost.cross_entropy(y, y_, 'cost')

    # Accuracy
    correct_prediction = tf.equal(y_op, y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train op
    batch_size = 128
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

    for epoch in range(n_epochs):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(
                X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epochs, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err; train_acc += ac; n_batch += 1
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(
                                        X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err; val_acc += ac; n_batch += 1
            print("   val loss: %f" % (val_loss / n_batch))
            print("   val acc: %f" % (val_acc / n_batch))

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(
                                X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one( network.all_drop )
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print("   test loss: %f" % (test_loss / n_batch))
    print("   test acc: %f" % (test_acc / n_batch))

    # In the end, close TensorFlow session.
    sess.close()
    tl.layers.clear_layers_name()
    tf.reset_default_graph()


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
    workers = ['node01', 'node02', 'node03', 'node04', 'node05']

    def submit_job(node_name, job_id):
        print('Assign job: {} to {}'.format(job_id, node_name))
        worker(job_id)

    # Submit jobs to all workers
    submit_job(workers[0], job_ids[0])
    submit_job(workers[2], job_ids[2])
    submit_job(workers[4], job_ids[4])


def master():
    db = TensorDB(ip='146.169.33.34', port=27020, db_name='TransferGan', user_name='akara', password='DSIGPUfour')
    create_mnist_dataset(db=db)
    create_jobs(db=db, job_name="cv_mnist", models_dict={
        "cnn": {
            "lr": [0.01, 0.001, 0.001],
            "n_cnn_layers": [1, 2, 2],
            "n_filters": [64, 128, 256],
            "n_epochs": [10, 10, 10],
        },
        "mlp": {
            "lr": [0.05, 0.0001],
            "n_layers": [1, 2],
            "n_epochs": [10, 10],
        }
    })
    start_workers(db=db)


### Workder node ###


def load_mnist_data(db, shape=(-1, 28, 28, 1)):
    data, f_id = db.find_one_params(args={'type': 'mnist_dataset'})
    if not data:
        raise Exception("MNIST dataset not found !!")
    X_train, y_train, X_val, y_val, X_test, y_test = data

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    X_train = X_train.reshape(shape)
    X_val = X_val.reshape(shape)
    X_test = X_test.reshape(shape)

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

    from bson.objectid import ObjectId
    job = db.find_one_job(args={'_id': ObjectId(job_id)})
    if job['model'] == 'cnn':
        train_cnn(db=db, n_cnn_layers=job['n_cnn_layers'], lr=job['lr'], n_epochs=job['n_epochs'])
    elif job['model'] == 'mlp':
        train_mlp(db=db, n_layers=job['n_layers'], lr=job['lr'], n_epochs=job['n_epochs'])


### Main ###


def main():
    master()


if __name__ == '__main__':
    main()
