# Intro to Deep Learning

Introductory Course to the field of Deep Learning, commonly known as Deep Neural Networks. Throughout this course, participants will delve into the theoretical foundations of Deep Learning models, understanding their mechanisms and potential applications. By the end of the program, attendees will have acquired the skills to construct and train models to address real-world problems.

* Instructor of DL: [Elmer E. Alanoca C.](https://github.com/eealanoca)

## Requirements

### Installing PyTorch (on Google Server)

Go to https://colab.research.google.com and sign in with your Google account. If you do not have a Google account you can create one. From there you can create a new notebook.

### Install using Conda (for local installation)

`pip3 install torch torchvision torchaudio`

## Course Organization

### 1. Fundamentals

Introduction, AI vs ML vs DL. Why DL now?

#### 1.1. Modern Neural Networks

* Perceptron, activation functions, and matrix representation.
* UAT, Feed-Forward Networks, and output function (softmax).
* Gradient Descent to find network parameters.
* Computational graphs and the BackPropagation algorithm.
* Tensors, Einstein Notation, and Tensor Chain Rule.
* Cross-Entropy and Backpropagation with Tensors.
* Practical aspects of training and FF Network in pytorch.

Readings: [Chapter 2. Linear Algebra](http://www.deeplearningbook.org/contents/linear_algebra.html), [Chapter 3. Probability and Information Theory](http://www.deeplearningbook.org/contents/prob.html), [Chapter 6. Deep Feedforward Networks](http://www.deeplearningbook.org/contents/mlp.html)

#### 1.2. Initialization, Regularization, and Optimization

* Generalization, Test-Dev-Train set, and Regularization.
* Ensemble, Dropout, and Gradient Vanishing.
* Parameter Initialization and Normalization.
* Optimization Algorithms, SGD with Momentum, RMSProp, Adam.

Readings: [Chapter 7. Regularization for Deep Learning](http://www.deeplearningbook.org/contents/regularization.html), [Chapter 8. Optimization for Training Deep Models](http://www.deeplearningbook.org/contents/optimization.html), [Chapter 11. Practical Methodology](http://www.deeplearningbook.org/contents/guidelines.html)

### 2. Convolutional Neural Networks (CNN)

* Introduction to Convolutional Networks.
* Well-known architectures: AlexNet, VGG, GoogLeNet, ResNet, DenseNet.

Readings: [Chapter 9. Convolutional Networks](http://www.deeplearningbook.org/contents/convnets.html), [Chapter 12. Applications](http://www.deeplearningbook.org/contents/applications.html)

### 3. Recurrent Neural Networks (RNN)

* Introduction to Recurrent Networks.
* Recurrent Network Architecture.
* Auto-regression, Language Modeling, and Seq-to-Seq Architectures.
* Gated and Memory Cell RNNs: GRU and LSTM.

Readings: [Chapter 10. Sequence Modeling: Recurrent and Recursive Nets](http://www.deeplearningbook.org/contents/rnn.html), [Chapter 12. Applications](http://www.deeplearningbook.org/contents/applications.html)

### 4. Advanced Topics

* Neural Attention.
* Transformers.
* Variational Autoencoders.
* Generative Adversarial Networks.
* Vision Transformers (ViT).

Readings: [Chapter 14. Autoencoders](http://www.deeplearningbook.org/contents/autoencoders.html), [Chapter 20. Deep Generative Models](http://www.deeplearningbook.org/contents/generative_models.html), [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929v2.pdf)

### 5. Reinforcement Learning

* Markov decision processes, returns, and policies.
* Markov process, Markov reward process, Markov decision process, Partially observable Markov decision process, Policy.
* Expected return.
* State and action values, Optimal policy, Bellman equations.
* Tabular reinforcement learning.
* Dynamic programming, Monte Carlo methods, Temporal difference methods. 
* Fitted Q-learning.
* Double Q-learning and double deep Q-network.
* Policy gradient methods.
* Derivation of gradient update REINFORCE algorithm, Baselines, State-dependent baselines.
* Actor-critic methods.
* Offline reinforcement learning.

Readings: [Chapters 2 - 13. Reinforcement Learning: An Introduction](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)

## Books

There is no mandatory textbook for the course. Some lectures will include suggested readings from "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville; however, it is not necessary to purchase a copy as it is available [online for free](http://www.deeplearningbook.org/).

1. [Deep Learning](http://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (fundamental bibliography for the course)
2. [Deep Learning for Vision Systems](https://livebook.manning.com/book/grokking-deep-learning-for-computer-vision/deep-learning-for-vision-systems/7) by Mohamed Elgendy
3. [Probabilistic and Statistical Models for Outlier Detection](https://www.springer.com/cda/content/document/cda_downloaddocument/9783319475776-c1.pdf?SGWID=0-0-45-1597574-p180317591) by Charu Aggarwal
4. [Pattern Recognition and Machine Learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) by Chistopher M. Bishop.
5. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf) by Daniel Jurafsky and James Martin
6. [Notes on Deep Learning for NLP](https://arxiv.org/abs/1808.09772) by Antoine J.-P. Tixier
7. [AutoML: Methods, Systems, Challenges](https://www.automl.org/book/) edited by Frank Hutter, Lars Kotthoff, and Joaquin Vanschoren
8. [Reinforcement Learning: An Introduction](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) by Richard S. Sutton and Andrew G. Barto.
##