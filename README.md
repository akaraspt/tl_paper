# Codes for ACM MM Open Source Software Competition

## Setup Environment
* TensorFlow 1.0
  * `pip2 install tensorflow-gpu`
* Install Database
  * Follow [MongoDB docs](https://docs.mongodb.com/manual/installation/)
  * We recommend to use one machine as database server.
* Python2 packages
  * `pip2 install pymongo gym numpy matplotlib scipy scikit-image`
  * `pi2 install gym[all]` or `pip install "gym[atari]"` (to install atari only)

## Raw Performance
* Run this experiment in solo mode.
* [TensorFlow Implementation](https://www.tensorflow.org/tutorials/deep_cnn)
* [TensorLayer Implementation](https://github.com/akaraspt/tl_paper/blob/master/cifar10.py), [Optimized Version](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cifar10_tfrecord.py) 5x faster, same accuracy, tested on Titan X Pascal

## Start Workers
* For `Deep Reinforcement Learning` and `Hyper-parameter selection and cross-validation` you may want to use distributed mode.
* Setup environment in all workers
* xx
* xx

## Deep Reinforcement Learning
* [tutorial_tensordb_atari_pong_generator.py](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_atari_pong_generator.py) is the data generator.
* [tutorial_tensordb_atari_pong_trainer.py](https://github.com/akaraspt/tl_paper/blob/master/tutorial_tensordb_atari_pong_trainer.py) is the GPU trainer.
* Run the experiment
  * To have a quick test, you can run multiple data generators and one trainer in a standalone machine.
  * For distributed mode.
    * Install MongoDB in one machine, setup environment in all other machines.
    * One machine run one trainer with GPU, all other machines run data generator.

## Hyper-parameter selection and cross-validation
* Hi Akara, if this part is in another repo, please link it

## Generative Adversarial Network
* Please check [Text to image synthesis](https://github.com/zsdonghao/text-to-image)

## More
* [Documentation](http://tensorlayer.readthedocs.io)
* [Examples and Applications](http://tensorlayer.readthedocs.io/en/latest/user/example.html)
