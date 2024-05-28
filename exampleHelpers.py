# Data/vector handeling and extra math
import numpy as np
import pandas as pd
import math

# Optimisation
import scipy
from scipy import optimize
import sklearn.metrics as metrics

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Error handeling
import warnings

# Time
import time
import datetime

# System/file interaction
import os
import pickle# https://www.pythoncentral.io/how-to-pickle-unpickle-tutorial/
import shelve# https://stackoverflow.com/questions/2960864/how-to-save-all-the-variables-in-the-current-python-session




def random_function(t=None, s=None):
    
    if t==None:
        t = max([np.random.normal(loc=2, scale=1), 0.1])
    
    time.sleep(t)
    
    if s!=None:
        total = 0
        for i in range(s):
            total = total + np.random.rand()
        
        t = t + s
    
    return t