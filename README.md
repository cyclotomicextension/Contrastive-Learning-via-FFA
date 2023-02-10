# Contrastive Learning via Forward-Forward algorithm

In the .py file attached, I have tried implementing a simple supervised example of FF via representation learning detailed in `The Forward-Forward Algorithm: Some Preliminary Investigations' by Geoffrey Hinton @ Google Brain. 
However, my implemention may not be an exact representation of what he has outlined. This repository and the code attached in this, has been implemented to learn more about CNN and Representation Learning using PyTorch.

# Introduction

The Forward-Forward (FF) algorithm is an unsupervised learning method that aims to learn multi-layer representations that capture the structure in the data. In this implementation, we will use FF to perform representation learning using real data vectors as the positive examples and corrupted data vectors as the negative examples.

# Prerequisites

A good source of negative data
PyTorch

# Data Preparation

To create the negative data, we use a mask containing regions of ones and zeros. The negative data is created by adding together one digit image multiplied by the mask and a different digit image multiplied by the reverse of the mask. Masks are created by starting with a random bit image and then repeatedly blurring the image with a filter of the form [1/4, 1/2, 1/4] in both the horizontal and vertical directions. After repeated blurring, the image is thresholded at 0.5.

# Implementation

The implementation of FF is done in PyTorch. To perform representation learning, we first learn to transform the input vectors into representation vectors without using any information about the labels. This is done by using real data vectors as the positive data and corrupted data vectors as the negative data. The negative data is created by transforming the original image data. The first step is to create a binary mask, where each element of the image is thresholded by 0.5. Elements with values greater than 0.5 are set to 1, while the rest are set to 0. The mask is then cast to float data type. We then create a hybrid image by adding together original image times the mask and image times the reverse of the mask. This hybrid image is then transformed into representation vectors, and a simple linear transformation maps these representation vectors to vectors of logits.

The learning of the linear transformation to the logits is supervised but does not involve learning any hidden layers so it does not require backpropagation of derivatives. 

# Usage

Clone the repository:

`git clone https://github.com/<your_username>/FF-algorithm.git`

Install the required packages:

`pip install -r requirements.txt`

Run the script:

`python main.py`

# Resources

<a href="https://discuss.pytorch.org/t/valueerror-expected-input-batch-size-324-to-match-target-batch-size-4/24498">ValueError: Expected input batch size 324 to match target batch size 4</a>

<a href="https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/">Writing CNNs from scratch in PyTorch</a>

<a href="https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device">RuntimeError: expected all tensors to be on the same device</a>

# Contributions

All contributions are welcome. If you find a bug, please submit an issue. If you want to contribute to the code, please submit a pull request. 
