# Neural Network Compression and Acceleration Methods

I am currently doing a survey of the different neural network 
compression and acceleration methods. In this survey, I will choose some 
of the most representative techniques of the different methods that can 
be implemented with the available frameworks.

This repo currently contains a PyTorch implementation of the popular 
VGG16 network, which will serve as a base for the network comparisons 
with the selected methods. The methods chosen will be added to this 
repo.

## Methods
We have chosen to implement VGGNet (16 layers) as the baseline model in order to compare and evaluate the different methods. We train our different implementations of the network on the popular benchmark dataset CIFAR-10. A slight modification was made to the the structure of the fully-connected layers of the network described in the paper. Since we are working with CIFAR-10 dataset (images 32 x 32 x 3) instead of ImageNet (images 224 x 224 x 3) as in the paper, we replaced the three fully-connected layers with one fully-connected layer of input size 1 x 1 x 512 (so, 512). This change has a sizeable impact in terms of number of parameters, but we should still be able to observe a difference between the different methods in comparison. Moreover, our training method uses a batch size of 128 instead of 256 since the dataset used is smaller than the original ImageNet.

## Training & Testing Info
| Dataset           | Dimension   | Labels   | Training set   | Test set   | Training epochs |
| ----------------- | ---------------------- | -------- | -------------- | ---------- | --------------- |
| CIFAR-10          | 3072 (32 x 32 color)   | 10   | 50K   | 10K   | 200 |

### Quantization

### Filter-level Pruning

### Compact Network Design
In order to have more comparable results, we chose to implement the VGG-16 architecture with depthwise separable convolutions. Our compact VGG-16 architecture, namely VGG16-DS in the tables below, is based on the [MobileNet](https://arxiv.org/abs/1704.04861) implementation, replacing all of the standard convolutions with depthwise separable convolutions except for the first layer which is a full convolution. Theoretically, this factorization has the effect of drastically reducing computation and model size. In practice, the efficiency of this method depends on the implementation within the framework used (here, PyTorch).

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)          | 90.03% |
| VGG16-DS									        | 86.55% |
| VGG16-Quant									    | - |
| VGG16-FPruned									    | - |
