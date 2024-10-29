# Machine Learning 2 Labs UvA

This repository contains lab solutions from the **Machine Learning 2 course at University of Amsterdam**, focusing on probabilistic graphical models and inference techniques. We explore advanced topics from Pattern Recognition and Machine Learning by C. Bishop, including:

 - Exponential families
 - Conditional independence
 - Information theory
 - Independent components analysis
 - Graphical models
 - Latent variable models
 - Learning, exact and approximate inference
 - Variational Inference
 - Sampling methods, MCMC, etc.
 - Sequential data models

 ## Installation

 The projects uses Poetry to manage dependencies. All the dependencies are in pyproject.toml. To install the them, run the following command:

```bash
poetry install
```

## Table of contents:

1) [Lab 1: Independent Component Analysis](src/Lab1/lab1.ipynb)

Implement algorithms to separate independent sources from observed mixed signals using techniques from information theory and statistical modeling.

2) [Lab 2: Inference in Graphical Models using Message Passing](src/Lab2/lab2.ipynb)

Developed inference methods for graphical models, utilizing message-passing algorithms to perform exact and approximate inference.

3) [Lab 3: State-space Models for Sequential Data](src/Lab3/lab3.ipynb)

Apply Hidden Markov Models (HMMs) and Kalman Filters for sequential data modeling, focusing on parameter learning and state estimation. Using Hidden Markov Model we tried to predict most optimal strategy to buy and sell Bitcoin.