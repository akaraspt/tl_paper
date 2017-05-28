# Codes for ACM MM Open Source Software Competition
This repo summarize the codes that mention in "TensorLayer: XXXXXXXX ⚠️".

## Set Up Environment
Before you start, you need to set up the environment for all examples in this repo.
* Layer Module
  * This is the basic module of TensorLayer, you can run all [single machine examples](http://tensorlayer.readthedocs.io/en/latest/user/example.html) with this module only.
  * TensorFlow 1.0+
    * `pip install tensorflow-gpu` or follow [the official installation guide](https://www.tensorflow.org/install/)
  * TensorLayer 
    * It is self-contained
    * You can also get the latest version in [Github](https://github.com/zsdonghao/tensorlayer)
* Deep Reinforcement Learning Environment
  * To run the DRL example, you will need to install [OpenAI gym](https://gym.openai.com) for game environment, [lz4](http://python-lz4.readthedocs.io/en/latest/quickstart.html) for data compression and the database module.
  * `pip install gym lz4`
  * `sudo apt-get install swig cmake`
  * `pip install gym[atari]` or `pip install gym[all]`
* Database and Workflow Module
  * For `Deep Reinforcement Learning` and `Hyper-parameter selection and cross-validation` you may want to use dataset and workflow modules.
  * Install MongoDB
    * Follow [MongoDB docs](https://docs.mongodb.com/manual/installation/)
    * We recommend to use one machine as database server.
  * Install eAE (Optional)
    * You may need this tool to distribute xxxx ⚠️

## Raw Performance
This a raw performance comparsion between TensorLayer and original TensorFlow engine, to prove TensorLayer's simplicity would not sacrifice the performance.
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
This is a simple asynchronous DRL example, you can run this example in one machine or multiple machines with database module.
* About the codes
  * [tutorial_tensordb_atari_pong_generator.py](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_atari_pong_generator.py) is the data generator.
  * [tutorial_tensordb_atari_pong_trainer.py](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_atari_pong_trainer.py) is the GPU trainer.
* Run the experiment
  * Before you run it in distributed mode, we higly recommend you to run one trainer with multiple data generators in a standalone machine.
  * For distributed mode, one machine run one trainer with GPU, all other machines run data generator.
    * Start
      * xxx
    * Monitor
      * xx
    * Terminate
      * xx

## Hyper-parameter selection and cross-validation
* xxx ⚠️Hi Akara, if this part is in another repo, please link it

## Generative Adversarial Network
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
