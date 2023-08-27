"""
File: preprocess.py
Description: This file contains any functions relating to importing and preprocessing the data from the MNIST dataset.

Author: Daniel Holmes
Created: 15/08/2023
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def preprocess():
    """
    This function imports and normalises the data from the MNIST dataset
    """

    # import the data from the mnist dataset using keras
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape the data to fit the model
    X_train = X_train.reshape(60000, 28, 28, 1)

    # normalise the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0
