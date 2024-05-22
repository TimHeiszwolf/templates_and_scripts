import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy import optimize
import sklearn.metrics as metrics

import warnings

## Data transformation
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
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ExpSineSquared
from sklearn.model_selection import train_test_split
def make_Gaussian_proccess_model(input, output, kernel=RBF()+WhiteKernel(), validation_fraction=0.75, make_plot=False):
    
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