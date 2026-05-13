# Neural Network From Scratch With NumPy

A fully connected neural network implemented entirely with NumPy without using deep learning frameworks such as TensorFlow or PyTorch.

## Features

- Forward propagation
- Backpropagation
- Gradient descent
- MNIST digit classification
- Training history visualization
- Neural network architecture visualization
- Prediction visualization
- Training animation (GIF)

## Project Structure

```text
src/
├── data_loader.py
├── neural_net.py
├── train.py

## Neural Network Architecture

Input Layer: 784 neurons  
Hidden Layer 1: 128 neurons  
Hidden Layer 2: 64 neurons  
Output Layer: 10 neurons

Activation:
- ReLU
- Softmax

Loss:
- Cross Entropy