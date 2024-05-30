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

from helpers import *


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

def calculate_pi_monte_carlo(samples, progress=False):
    """
    With this function you can calculate pi. It does this based on the area of a circle being pi*r^2 while a square is (2r)^2. So a quadrant then is r^2 and pi/4 r^2. Thus when placing random dots the ratio between the total amount of dots and the amount of dots within the unit circle is pi/4.
    """
    samples = samples + 1
    x = np.random.rand(samples)
    y = np.random.rand(samples)
    amount_within_circle = 0
    amount_outside_circle = 0

    start_time = time.time()### IMPORTANT
    last_update = time.time()### IMPORTANT
    for i in range(samples):
        length = np.sqrt(x[i]**2+y[i]**2)
        if length<=1:
            amount_within_circle = amount_within_circle + 1
        else:
            amount_outside_circle = amount_outside_circle + 1
        calculated_pi = 4*amount_within_circle/(amount_within_circle + amount_outside_circle)

        if progress:
            last_update = update_progress(i/(samples-1), bar_length=50, start_time=start_time, message="Current pi: "+str([calculated_pi, i]), last_update_time=last_update, refresh_rate=5)### IMPORTANT
    
    return calculated_pi