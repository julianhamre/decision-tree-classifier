import numpy as np

def numerical_threshold(values):
    return np.mean(values)

def categorical_threshold(values):
    return np.min(values)
