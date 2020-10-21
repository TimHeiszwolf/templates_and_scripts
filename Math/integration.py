def integration_Simpson(function, integration_range, number_of_intervals = 1000):
    """
    A fucntion for intergration of a 1D function. This function stolen from example documents of the course 3FX0 Physical modelling and simulating of the TU/e.
    """
    x = np.linspace(integration_range[0], integration_range[1] , number_of_intervals + 1)
    y = function(x)
    h = x[1] - x[0]
    
    n = len(y) - 1
    s0 = 0
    s1 = 0
    s2 = 0
    for i in range(1, n, 2):
        s0 += y[i]
        s1 += y[i - 1]
        s2 += y[i + 1]
    s = (s1 + 4 * s0 + s2) / 3
    if (n + 1)%2 == 0:
        return h * (s + (5 * y[n] + 8 * y[n - 1] - y[n - 2]) / 12);
    else:
        return h * s

 
def integration_monte_carlo_random(function, amount_of_points, boundary = [[0, 1]]):
    """
    A function which can do a Monte Carlo integral using random sampeling.
    
    function is the function which get's integrated. The input should be a numpy array.
    amount_of_points is the total amount of points used in the calculation.
    boundary is the range of the input. Starts as a simple integration from 0 to 1 in 1D.    
    """
    total = 0
    total_error = 0

    for j in range(amount_of_points):
        x = np.array([random.uniform(boundary[i][0], boundary[i][1]) for i in range(len(boundary))])
        result = function(x)
        total = total + result
        total_error = total_error + result**2
        
    return total / amount_of_points, np.sqrt(abs((total_error / amount_of_points - (total / amount_of_points)**2) / amount_of_points))


def integration_rieman(function, step_size = 0.01, dimensions = 1, current_boundary = [[0, 1]], start_position = []):
    """
    A function which does a N dimensional Rieman integration using recursion. Currently using midpoint.
    
    function is the function which is going to be integrated. It needs to be able to take a vector.
    step_size is the size of the steps taken in the integration. It is imposibble to select different step sizes for different dimensions.
    dimensions is the dimensions of the space to be integrated.
    current_boundary is the current boundairy. It is a list consisting of lists. The outer list is for each dimension and the inner list is for the boundairy in the dimension itself
    """
    current_total = 0
    midpoint = step_size / 2
    
    if len(current_boundary) == 1:
        boundary = current_boundary[0]
        size = boundary[1] - boundary[0]
        
        dx = (step_size**(dimensions))
        
        for step in range(math.ceil(size / step_size)):
            position = start_position.copy()
            position.append(boundary[0] + step_size * step + midpoint)
            
            current_total = current_total +  function(position) * dx
            # print(position, function(position), dx * function(position))
        
        return current_total
    else:
        boundary = current_boundary[0]
        size = boundary[1] - boundary[0]
        new_boundairy = [current_boundary[i] for i in range(1, len(current_boundary))]
        
        for step in range(math.ceil(size / step_size)):
            position = start_position.copy()
            position.append(boundary[0] + step_size * step + midpoint)
            
            current_total = current_total + normal_integration(function, step_size, dimensions, new_boundairy, position)
        
        return current_total