# Codes for ACM MM Open Source Software Competition

## Setup Environment
* TensorFlow 1.0
  * `pip install tensorflow-gpu`
* TensorLayer 
  * It is self-contained
  * You can also get the latest version in [Github](https://github.com/zsdonghao/tensorlayer)
* Install Database
  * Follow [MongoDB docs](https://docs.mongodb.com/manual/installation/)
  * We recommend to use one machine as database server.
* Python packages
  * `pip install pymongo gym numpy matplotlib scipy scikit-image lz4`
  * `sudo apt-get install swig cmake`
  * `pip install gym[atari]` or `pip install gym[all]` 

## Raw Performance
* Run this experiment in solo mode.
* [TensorFlow Implementation](https://www.tensorflow.org/tutorials/deep_cnn)
* [TensorLayer Implementation](https://github.com/akaraspt/tl_paper/blob/master/cifar10.py), [Optimized Version](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cifar10_tfrecord.py) (5x faster, optimize CPU/GPU operation, tested on Titan X Pascal)

## Start Workers
For `Deep Reinforcement Learning` and `Hyper-parameter selection and cross-validation` you may want to use distributed mode.
* Install MongoDB in one machine, setup environment in all other machines.
* xx
* xx

## Deep Reinforcement Learning
* [tutorial_tensordb_atari_pong_generator.py](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_atari_pong_generator.py) is the data generator.
* [tutorial_tensordb_atari_pong_trainer.py](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_atari_pong_trainer.py) is the GPU trainer.
* Run the experiment
  * To have a quick test, you can run multiple data generators and one trainer in a standalone machine.
  * For distributed mode, one machine run one trainer with GPU, all other machines run data generator.

## Hyper-parameter selection and cross-validation
* Hi Akara, if this part is in another repo, please link it

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
