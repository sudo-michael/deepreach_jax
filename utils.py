import numpy as np
import os

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def unormalize_value_function(x, norm_to, mean, var):
    return (x * var / norm_to) + mean

def normalize_value_function(x, norm_to, mean, var):
    return (x - mean) * norm_to / var