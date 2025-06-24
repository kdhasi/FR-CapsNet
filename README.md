# FR-CapsNet

Pytorch implementation of the paper "FR-CapsNet: Enhancing Low-Resolution Image Classification via Frequency Routed Capsules." The manuscript is available at XX.

## Prerequisites

- Python 3.9
- CUDA 11.8

Install the required packages by:

 ```
pip install -r requirements.txt
 ```

## Train

To train a model run:

 ```
python main.py --dataset=[cifar10] --name=resnet_frequency_routing --epochs=350 --is_train=True
 ```

```dataset``` should be one of [mnist, svhn, cifar10, cifar100]

## Test

To test a model run:

 ```
python main.py --dataset=[cifar10] --name=resnet_frequency_routing --epochs=350 --is_train=False
 ```

To perform adversarial attacks against a trained model run:

 ```
python main.py --dataset=[cifar10] --name=resnet_frequency_routing --epochs=350 --is_train=False --attack=True --attack_type=bim --attack_eps=0.1 --targeted=False
 ```

## Datasets

- MNIST
- SVHN
- CIFAR-10
- CIFAR-100

## Citation

If you find this work useful, please cite the following paper:
