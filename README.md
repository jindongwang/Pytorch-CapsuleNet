# Pytorch-CapsuleNet

A flexible and easy-to-follow Pytorch implementation of Hinton's Capsule Network.

There are already many repos containing the code for CapsNet. However, most of them are too tight to customize. And as we all know, Hinton's original paper is only tested on *MNIST* datasets. We clearly want to do more.

This repo is designed to hold other datasets and configurations. And the most important thing is, we want to make the code **flexible**. Then, we can *tailor* the network according to our needs.

Currently, the code supports both **MNIST and CIFAR-10** datasets.

## Requirements

- Python 3.x
- Pytorch 0.3.0 or above
- Numpy
- tqdm (to make display better, of course you can replace it with 'print')

## Run

Just run `Python test_capsnet.py` in your terminal. That's all. If you want to change the dataset (MNIST or CIFAR-10), you can easily set the `dataset` variable.

It is better to run the code on a server with GPUs. Capsule network demands good computing devices. For instance, on my device (Nvidia K80), it will take about 5 minutes for one epoch of the MNIST datasets (batch size = 100).

## More details

There are 3 `.py` files:
- `capsnet.py`: the main class for capsule network
- `data_loader.py`: the class to hold many classes
- `test_capsnet.py`: the training and testing code

The results on your device may look like the following picture:

![](https://raw.githubusercontent.com/jindongwang/Pytorch-CapsuleNet/master/result.jpg)

## Acknowledgements

- [Capsule-Network-Tutorial](https://github.com/higgsfield/Capsule-Network-Tutorial)
- The original paper of Capsule Net by Geoffrey Hinton: [Dynamic routing between capsules](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules)
