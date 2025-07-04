# Driver-Drowsiness-Detection-Model
# Eye State Classification with PyTorch CNN
This repository contains the code and results for a Convolutional Neural Network (CNN) model developed using PyTorch to classify eye states as either "open" or "closed".

# Project Overview
The goal of this project was to build and train a deep learning model capable of accurately distinguishing between images of open and closed eyes. This type of classification has applications in various fields, including driver drowsiness detection, human-computer interaction, and accessibility features.

# Dataset
The model was trained and evaluated on the MRL Eye Dataset, specifically from the /kaggle/input/mrl-dataset/train directory. The dataset is structured with separate folders for "openeyes" and "closedeyes," allowing torchvision.datasets.ImageFolder to automatically infer class labels.

# Data Split:

Training Set: 80% of the dataset

Testing Set: 20% of the dataset

# Image Preprocessing:
Images were transformed using:

Resizing to 256x256 pixels.

Random horizontal flips (for training data).

Random rotations (for training data).

Conversion to PyTorch Tensors.

Normalization using standard ImageNet mean and standard deviation.

# Model Architecture
The model, named DDDM, is a custom-built Convolutional Neural Network implemented using torch.nn.Module. It consists of:

Two Convolutional Blocks: Each block comprises:

nn.Conv2d: For feature extraction.

nn.BatchNorm2d: For stabilizing and accelerating training.

nn.ReLU: As the activation function, introducing non-linearity.

nn.MaxPool2d: For spatial downsampling.

Flatten Layer: To convert the 2D feature maps into a 1D vector.

Two Fully Connected (Linear) Layers:

A hidden linear layer with 128 output features.

An output linear layer that maps features to the 2 classification classes (open/closed).

Training and Evaluation
The model was trained for 10 epochs using the following configuration:

Loss Function: nn.CrossEntropyLoss (suitable for multi-class classification).

Optimizer: Adam with a learning rate of 0.001.

Device: Training was performed on a GPU (cuda) if available, otherwise on CPU.

# Results
After training, the model achieved the following performance on the test set:

Test Loss: 0.0388

Test Accuracy: 99.12%

These results indicate that the model performs exceptionally well in classifying open versus closed eyes on the given dataset.

# Conclusion & Next Steps
The high accuracy of 99.12% demonstrates the effectiveness of the chosen CNN architecture and training approach for this specific binary classification task. The clear visual distinction between open and closed eyes in the dataset likely contributed to this strong performance.

To further enhance the model's robustness and real-world applicability, future work could include:

Dataset Expansion: Incorporating more diverse images with varying lighting conditions, facial expressions, occlusions (e.g., glasses, hair), and different individuals.

Generalization Testing: Evaluating the model on entirely new, unseen datasets to confirm its ability to generalize beyond the training distribution.

Hyperparameter Tuning: Experimenting with different learning rates, batch sizes, optimizer types, and network architectures to potentially achieve even better or more stable performance.

Deployment Considerations: Exploring methods for deploying the model in real-time applications, such as optimizing for mobile or embedded devices.
