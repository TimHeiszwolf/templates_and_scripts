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


### Data transformation
def transformToGrid(x, y, z, xName="x", yName="y", zName="z", xn=False, yn=False):
    """
    Transforms 3 arrays of (corresponding) x, y, z values into a grid. For example for use in a heat-maps, surface plots, contour plots, etc.

    x, y, and z are (sequential/1D) arrays which contain the x and y coordinated and the coresponding z coordiante for each index. 

    Returns grids (like a meshgrid) for each coordinate and (pivoted) dataframe.
    """
    if xn==False and yn==False:#If xn and yn are not given assume it is a perfect grid
        xn = len(set(x))
        yn = len(set(y))
    
    xGrid = np.zeros([xn, yn])
    yGrid = np.zeros([xn, yn])
    zGrid = np.zeros([xn, yn])
    pdDataframe = pd.DataFrame({xName:x, yName:y, zName:z})
    
    k=0
    for i in range(xn):# TODO Maybe also do this with a panda DataFrame.
        for j in range(yn):
            xGrid[i, j] = x[k]
            yGrid[i, j] = y[k]
            zGrid[i, j] = z[k]
            k=k+1
    
    return xGrid, yGrid, zGrid, pdDataframe.pivot(index=yName, columns=xName)[zName]

### Performance
## Progress bar
#from IPython.display import clear_output
def update_progress(progress:float, bar_length=50, start_time=None, message=None, last_update_time=0, refresh_rate=2):
    """
    Generates a progress bar and is based on the float progress which should be between 0 and 1. 
    Can also display and make linear time estimations. You can also have it refresh at a certain rate. If you don't want that then leave last_update_time at zero.
    
    Inspired by: https://mikulskibartosz.name/how-to-display-a-progress-bar-in-jupyter-notebook-47bd4c2944bf
    """
    current_time = time.time()
    
    if type(progress)!=float:
        raise TypeError("Progress is not an float but an " + str(type(progress)))
    elif progress<0 or progress>1:
        warnings.warn("Progress is not between 0 and 1 but is " + str(progress))
        progress = max([min([progress, 1]), 0])# Ensure the value of progress is beteween 0 and 1.
    
    if (current_time-last_update_time)>(1/refresh_rate) or progress==1:# Limit the update_rate
        block = int(round(bar_length * progress))
        text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
        
        if start_time!=None:
            
            if (current_time - start_time)<0:
                warnings.warn("Warning: time difference is progress bar is negative: "+str(current_time - start_time))
            
            progress_per_time = progress/max([current_time - start_time, 10**-20])# Ensure a minimum to prevent 1/0 error.
            expected_remaining_time = (1 - progress)/max([progress_per_time, 10**-20])# Ensure a minimum to prevent 1/0 error.
            
            if progress_per_time > 0.01:
                time_unit = "sec"
            elif progress_per_time > 0.01/60:
                time_unit = "min"
                progress_per_time = progress_per_time*60
                expected_remaining_time = expected_remaining_time/60
            elif progress_per_time > 0.01/3600:
                time_unit = "hour"
                progress_per_time = progress_per_time*3600
                expected_remaining_time = expected_remaining_time/3600
            else:# progress_per_time > 0.01/86400:
                time_unit = "day"
                progress_per_time = progress_per_time*86400
                expected_remaining_time = expected_remaining_time/86400
            
            text = text + " | "  + str(np.round(100*progress_per_time, 2)) + " %/" + time_unit + " | " + str(np.round(expected_remaining_time, 1)) + " " + time_unit + " remaining"
        
        if message!=None:
            text = text + " | " + message
        
        if progress!=1:
            print(text.ljust(len(text)*5), end="\r")#https://stackoverflow.com/questions/5290994/remove-and-replace-printed-items
        else:
            print(text.ljust(len(text)*5))# Pad the text with a great amount of with space to prevent characters being left over from previous progress print.
        
        return current_time# Return the current time if an update occured to thus update the progress bar.
    else:
        return last_update_time

### Fitting
## Polynomial fitting
def polynomialFuction(x, coefficients):
    """
    Calculates an arbetrairy order polynomial of the order of the length of coefficients. Coefficients are the coefficients of each term (starting at 0th order).
    """
    return np.sum([coefficients[i] * x**i for i in range(len(coefficients))])


def makePolynomialFit(xData, yData, displayResults=False, maxOrderFraction=1/5, largestMAPERatio=1.1, warningMAPE=50):
    """
    Fits an (semi-)arbitrary polynomial function to the data and returns the coefficients (the first element is order 0, second order 1, etc) and MAPE, can also print and graph some of the results.
    
    xData is a numpy-list of the x-axis data points.
    yData is a numpy-list of the y-axis data points.
    displayResults determines if the results are printed and plotted.
    maxOrderFraction is used to determine the maximum order to be fitted (as a fraction of the total amount of data points).
    largestMAPERatio is the largest possible MAPE ratio between the selected polynomial (preferable an order as low as possible to prevent overfitting) and the polynomial with the lowest MAPE.
    warningMAPE is the MAPE at which the function will raise a warning if the final fit has a higher MAPE than it.
    """
    
    maxOrder = int(np.floor(len(xData)*maxOrderFraction))# The largest order which is to be fitted.
    
    MAPEs = []
    for i in range(maxOrder + 1):
        coefficients = np.polynomial.polynomial.Polynomial.fit(xData, yData, i).convert().coef# Make the fit and get the coefficients
        fit = [polynomialFuction(x, coefficients) for x in xData]# Get the values of the fit.
        MAPE = 100*metrics.mean_absolute_percentage_error(yData, fit)
        MAPEs.append(MAPE)
        
        #print(i, "MAPE:", MAPE, "Coef:", coefficients)
        #print(np.polynomial.polynomial.Polynomial.fit(xData, yData, i, full=True)[1][0])
    
    minimumMAPE = min(MAPEs)
    
    for i in range(maxOrder + 1):# Select the polynomial to be used by starting at the lowest order going up until a polynomial can be found which has a low enough MAPE.
        if minimumMAPE*largestMAPERatio > MAPEs[i]:
            selectedOrder = i
            break
    
    coefficients = np.polynomial.polynomial.Polynomial.fit(xData, yData, selectedOrder).convert().coef# Redo the fit since we didn't save all the fits.
    fit = [polynomialFuction(x, coefficients) for x in xData]
    MAPE = 100*metrics.mean_absolute_percentage_error(yData, fit)
    
    if MAPE>=warningMAPE:# Detect if the total result is good enough and else raise a warning.
        warnings.warn("Warning: high MAPE ("+str(np.round(MAPE))+"). A good fit was likely not found, possibly due to the order not being high enough, not enough data points or the data not being smooth enough/growing too hard.")
    
    if displayResults:# If desired print the results and make a plot of the data and fit.
        print("Selected order: order", selectedOrder, "of", maxOrder, ", MAPE:", MAPEs[i], ", minimum MAPE:", minimumMAPE)
        print("Coefficients:", coefficients)
        fig, ax = plt.subplots(1, 1, figsize = (6,6))# Make a plot.
        ax.scatter(xData, yData, label="data")
        ax.plot(xData, fit, "--", c="orange", label="fit")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Polynomial fit (order "+str(selectedOrder)+") on the data")
        ax.legend()
        plt.show()
    
    return coefficients, MAPE

## Gaussian procces fitting
from sklearn.gaussian_process import GaussianProcessRegressor# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ExpSineSquared, PairwiseKernel
from sklearn.model_selection import train_test_split

def make_Gaussian_proccess_model(input, output, kernel=RBF()+WhiteKernel(), validation_fraction=0.25, make_plot=False):
    """
    A function that makes a guassian procces model based on some data. An advanced way of fitting usefull in certain situations.
    """
    if len(np.shape(input))==1:# If it is a simple 1D vector transform it into a two layered vector.
        input = input[:,None]
    elif np.shape(input)[0]>1:
        make_plot=False
        warnings.warn("Can't make plot with more than 1D input.")
        
    x_train, x_val, y_train, y_val = train_test_split(input, output, test_size=validation_fraction)
    
    regressor = GaussianProcessRegressor(kernel, n_restarts_optimizer=10)
    regressor.fit(x_train, y_train)
    
    y_train_pred, y_train_pred_std = regressor.predict(x_train, return_std=True)
    y_val_pred, y_val_pred_std = regressor.predict(x_val, return_std=True)
    
    training_error = {"RMSE":metrics.mean_squared_error(y_train, y_train_pred, squared=False),"MAPE": 100*metrics.mean_absolute_percentage_error(y_train, y_train_pred), "MAE": metrics.mean_absolute_error(y_train, y_train_pred), "MSE": metrics.mean_squared_error(y_train, y_train_pred), "R2": metrics.r2_score(y_train, y_train_pred)}
    validation_error = {"RMSE":metrics.mean_squared_error(y_val, y_val_pred, squared=False), "MAPE": 100*metrics.mean_absolute_percentage_error(y_val, y_val_pred), "MAE": metrics.mean_absolute_error(y_val, y_val_pred), "MSE": metrics.mean_squared_error(y_val, y_val_pred), "R2": metrics.r2_score(y_val, y_val_pred)}
    
    if make_plot:
        x_pred = np.linspace(np.min(input)-np.abs(np.min(input)*0.25), np.max(input)+np.abs(np.max(input)*0.25))
        y_pred, y_pred_std = regressor.predict(x_pred[:,None], return_std=True)
        
        #sigma_e = np.exp(regression.kernel_.k2.theta)**0.5# extract the noise
        #y_pred_std = (y_pred_std**2 - sigma_e**2)**0.5# remove sigma_e noise to get mean deviation

        fig, ax = plt.subplots(1, 1, figsize = (6,6))
        ax.scatter(x_train, y_train, label="Training data")
        if len(x_val)>0:
            ax.scatter(x_val, y_val, label="Validation data")
        ax.plot(x_pred, y_pred, linestyle="--", label="Fit")
        ax.fill_between(x_pred, y_pred-2*y_pred_std, y_pred+2*y_pred_std, alpha=0.3, label="95% interval")
        ax.legend()
        plt.show()
        
        print("Training error:", training_error)
        print("Validation error:", validation_error)
    
    return regressor, training_error, validation_error

## (Torch) Neural Networks
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

def train_neural_net_torch(model, x_train, y_train, epochs, validation_fraction=0.3, batch_size=10, loss_function=nn.MSELoss(), optimizer=None, early_stopping_threshold=None, verbose=1):
    """
    A function which can train a (py)torch neural network. Based on this tutorial. https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/. For basic neural nets this works fine, but for larger, more complex needs I would use purpuse build functions and/or Keras.
    """
    history = {"epoch":[],"loss":[]}
    if validation_fraction>0:
        history["val_loss"]= []
    
    if optimizer==None:
        optimizer= optim.Adam(model.parameters(), lr=0.001)
    
    if early_stopping_threshold==None:# If early stopping is not configured make it so early stopping can never happen.
        early_stopping_threshold = 2*epochs
    
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_fraction)# Split of 30% to use for validation.
    
    best_score = np.inf
    best_epoch = 1
    
    start_time = time.time()# Both are used for the progress bar.
    last_update_time = time.time()
    
    epoch = 0# Variable used in the while loop.
    while epoch<epochs:
        epoch = epoch + 1
        for i in range(0, len(x_train), batch_size):
            # Training
            x_batch = x_train[i:i+batch_size]
            y_predict = model(x_batch)
            y_batch = y_train[i:i+batch_size]
            loss = loss_function(y_predict, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # There is no need to do validation during batch training.
            #if validation_fraction>0:
            #    x_batch = x_validation[i:i+batch_size]
            #    y_predict = model(x_batch)
            #    y_batch = y_validation[i:i+batch_size]
            #    loss = loss_function(y_predict, y_batch)
        
        # Logging relevant data/history
        history["epoch"].append(epoch)
        y_predict = model(x_train)
        loss = loss_function(y_predict, y_train).item()
        history["loss"].append(loss)
        message = "Epoch: " + str(epoch) + ", Loss: " + str(loss)
        
        if validation_fraction>0:
            y_predict = model(x_validation)
            loss = loss_function(y_predict, y_validation).item()
            history["val_loss"].append(loss)
            message = message + ", Validation loss: " + str(loss)
            
            if loss < best_score:# Keep track of the best epoch and score
                best_epoch = epoch
                best_score = loss
            
            if (epoch - best_epoch) > early_stopping_threshold:
                message = message + " | EARLY STOPPING AT EPOCH: " + str(epoch)
                epoch = epochs# Increase the epoch variable so high that the early stopping happens.

        if verbose>=1:# Give basic progress update with message if verbose is enabled.
            last_update_time = update_progress(epoch/epochs, bar_length=50, start_time=start_time, message=message, last_update_time=last_update_time, refresh_rate=2)
    
    if verbose>=2:# Make some nice plots if verbose is high enough.
        fig, ax1 = plt.subplots(1, 1, figsize = (6,6))# Make a plot.
        ax1.plot(history["epoch"], history["loss"], label="Training")
        ax1.plot(history["epoch"], history["val_loss"], label="Validation")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_yscale('log')
        ax1.set_title("Loss per epoch")
        ax1.legend()
        plt.show()
    
    return history