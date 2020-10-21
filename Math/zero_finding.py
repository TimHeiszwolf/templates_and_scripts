# These are implimentations of algorithms for finding zeros in functions. Numpy or something likely has functions for this but these are easy.
import warning


def find_zero_bisection(function, search_range = [-10, 10], offset = 0, minimum_accuracy = 0.1, max_steps = 10000):
    """
    Find a zero of a 1D function with the bisection method.
    
    function is the 1D function of which the zero needs to be found.
    search_range is the initial range in which the zero needs to be searched.
    offset is any potential offset (for if you don't want to find function(x)=0 but for example function(x)=offset.
    minimum_accuracy is the accuracy which needs to be reached before the algorithms stops.
    max_steps is the maximum amount of steps that the algorithm takes to find the zero. If it fails is raises a warning.
    
    """
    steps = 0
    delta = search_range[1] - search_range[0]
    
    while delta > minimum_accuracy:
        middle_value = np.mean(search_range)
        
        if ((function(search_range[0]) - offset) * (function(middle_value) - offset)) < 0:
            search_range[1] = middle_value
            delta = search_range[1] - search_range[0]
        else:
            search_range[0] = middle_value
            delta = search_range[1] - search_range[0]
        
        steps = steps + 1
        
        if steps >= max_steps:
            warning.warn('Didn\'t find the zero (with the required accuracy) within the maximum amount of steps/')
            break
    
    return np.mean(search_range), delta, steps, search_range
