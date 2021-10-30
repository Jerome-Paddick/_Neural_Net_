import math
import random
from random import randrange, random
import numpy as np


def activation_function(number):
    # logistic function
    return 1/(1 + np.exp(-number))


def sigmoid_neuron(inputs: np.array, weights: np.array):
    print(inputs.dot(weights))
    return activation_function(inputs.dot(weights))


def perceptron(inputs: np.array, weights: np.array, bias):
    print(inputs.dot(weights))
    if inputs.dot(weights) < bias:
        return False
    else:
        return True


example_inputs = np.array([randrange(10) for _ in range(10)])
example_weights = np.array([random()/5 for _ in range(10)])
bias = 5

print(example_inputs)
print(example_weights)
print(perceptron(example_inputs, example_weights, bias))
print(perceptron(example_inputs, 2*example_weights, 2*bias))