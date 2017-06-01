# Codes for ACM MM Open Source Software Competition
This repo summarizes the codes that mention in "TensorLayer: A Versatile Library for Efficient Deep Learning Development".

## Set Up Environment
Before you start, you need to set up the environment for all examples in this repo.
* Layer and Model Modules
  * These are the basic modules of TensorLayer, you can run all [single machine examples](http://tensorlayer.readthedocs.io/en/latest/user/example.html) with this module only.
  * TensorFlow 1.0+
    * `pip install tensorflow-gpu` or follow [the official installation guide](https://www.tensorflow.org/install/)
  * TensorLayer 
    * It is self-contained
    * You can also get the latest version in [Github](https://github.com/zsdonghao/tensorlayer)
* Deep Reinforcement Learning Environment
  * To run the DRL example, you will need to install [OpenAI gym](https://gym.openai.com) for game environment, [lz4](http://python-lz4.readthedocs.io/en/latest/quickstart.html) for data compression and the dataset module.
  * `pip install gym lz4`
  * `sudo apt-get install swig cmake`
  * `pip install gym[atari]` or `pip install gym[all]`
* Dataset and Workflow Modules
  * For `Deep Reinforcement Learning` and `Hyper-parameter selection and cross-validation` you may want to use dataset and workflow modules.
  * Install MongoDB
    * Follow [MongoDB docs](https://docs.mongodb.com/manual/installation/)
    * We recommend to use one machine as dataset server.
  * Install eAE (Optional)
    * You may need this environement to distribute the different jobs. An installation process is available at that address: https://github.com/aoehmichen/eae-docker/blob/master/install_eae_hybrid.txt 

## Raw Performance
You only need to install TensorFlow to run these examples. This a raw performance comparsion between TensorLayer and original TensorFlow engine, to prove TensorLayer's simplicity would not sacrifice the performance.
* CIFAR-10 classification
  * [TensorFlow Implementation](https://www.tensorflow.org/tutorials/deep_cnn)
  * [TensorLayer Implementation](https://github.com/akaraspt/tl_paper/blob/master/cifar10.py), [Optimized Version](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cifar10_tfrecord.py) (5x faster, optimize CPU/GPU operation, tested on Titan X Pascal)
* PTB language modelling
  * [TensorFlow Implementation](https://www.tensorflow.org/tutorials/recurrent)
  * [TensorLayer Implementation](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py)
* Word2vec
  * [TensorFlow Implementation](https://www.tensorflow.org/tutorials/word2vec)
  * [TensorLayer Implementation](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_word2vec_basic.py)

## Deep Reinforcement Learning
You need to install all modules to run this example.
This is a simple asynchronous DRL example, you can run this example in one machine or multiple machines with dataset module.
* About the codes
  * [tutorial_tensordb_atari_pong_generator.py](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_atari_pong_generator.py) is the data generator.
  * [tutorial_tensordb_atari_pong_trainer.py](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_atari_pong_trainer.py) is the GPU trainer.
* Run the experiment
  * Before you run it in distributed mode, we higly recommend you to run one trainer with multiple data generators in a standalone machine.
  * For distributed mode, one machine run one trainer with GPU, all other machines run data generator.
    * Start multiple agents that generate training samples for the trainer. Run the following script uses to start multiple generators. Note: all of the generated data will be stored in MongoDB, which will be fetched by the trainer
      *  `python submit_job_eae.py`
    * After start the generators, run the following script to train a model.
      * `python tutorial_tensordb_atari_pong_trainer.py`
    * To terminate all of the generators, run the following scripts.
      * `python terminate_running_jobs.py`

## Hyper-parameter selection and cross-validation
You need to install all modules to run this example.
* [DeepSleepNet](https://github.com/akaraspt/deepsleepnet)

## Generative Adversarial Network
You only need to install TensorFlow to run these examples.
* DCGAN
  * [TensorFlow Implementation](https://github.com/carpedm20/DCGAN-tensorflow)
  * [TensorLayer Implementation](https://github.com/zsdonghao/dcgan)
* Text to image synthesis
  * [TensorFlow Implementation](https://github.com/paarthneekhara/text-to-image)
  * [TensorLayer Implementation](https://github.com/zsdonghao/text-to-image)

  
## More
* [Documentation](http://tensorlayer.readthedocs.io)
* [Examples and Applications](http://tensorlayer.readthedocs.io/en/latest/user/example.html)
* [Github](https://github.com/zsdonghao/tensorlayer)
