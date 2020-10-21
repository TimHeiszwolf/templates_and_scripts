def ODE_euler_method(start_values, change_function, delta_time, end_time, start_time = 0):
    """
    A implimentation fo the Euler method for solving Ordinairy differential equations.
    
    start_values are the initial conditions of values array.
    change_function is the function which produces the derivates of the values.
    delta_time is the time-steps used for the Euler method.
    end_time is the end time of the simulation.
    start_time is the start time of the simulation.
    """
    time = start_time
    values = start_values
    
    data ={'time':[time], 'values':[values], 'derivative':[]}
    
    for i in range(1, math.ceil(end_time / delta_time) + 1):
        derivative = change_function(values)
        values = values + delta_time * derivative
        time = time + delta_time
        
        data['time'].append(time)
        data['values'].append(values)
        data['derivative'].append(derivative)
    
    data['derivative'].append(change_function(values))
    
    return data


def change_function(inputs, parameters = {'alpha':0.8, 'beta':1.2, 'gamma':0.4}):
    """
    An example of a change function for the ODE_euler_method.
    """
    animal_1 = inputs[0]
    animal_2 = inputs[1]
    
    alpha = parameters['alpha']
    beta = parameters['beta']
    gamma = parameters['gamma']
    
    delta_animal_1 = (alpha - gamma * animal_2) * animal_1
    delta_animal_2 = (gamma * animal_1 - beta) * animal_2
    
    return np.array([delta_animal_1, delta_animal_2])