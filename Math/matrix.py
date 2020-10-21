# Methods for dealing with Matrices. Don't use these. Just use the numpy functions.

def get_random_matrix(size = [5, 5], value_range = [-10, 10], type_of_value = int):
    """
    A function which can generate many different types of random matrixes of whatever dimension is desired.
    """
    if len(size)>1:
        current_size = size[0]
        new_size = [size[i] for i in range(1, len(size))]
        return [get_random_matrix(new_size, value_range, type_of_value) for i in range(0, current_size)]
    else:
        if type_of_value == int:
            return [random.randint(value_range[0], value_range[1]) for i in range(0, size[0])]
        if type_of_value == float:
            return [random.uniform(value_range[0], value_range[1]) for i in range(0, size[0])]
        else:
            print('Type:', type_of_value, 'is not supported by this function.')


def calculate_determinant(matrix):
    """
    A function which calculates the determinant of matrixes with recursion.
    
    TODO: add validation.
    """
    #print(np.array(matrix))
    size_matrix = len(matrix)
    if size_matrix==1:
        return matrix[0][0]
    else:
        total = 0
        
        for i in range(0, size_matrix):
            new_matrix = [[matrix[y][x] for x in range(0, size_matrix) if x != i] for y in range(1, size_matrix)]
            total = total + (-1)**i * matrix[0][i] * calculate_determinant(new_matrix)
        
        return total


def Guassian_elimination(matrix, vector):
    """
    https://www.youtube.com/watch?v=3aO2eG9lGk4
    """
    
    #print('ORG\n', matrix, vector, '\n')
    for i in range(0, min(len(matrix[0]), len(matrix))):
        max_factor = abs(matrix[i][i])
        max_factor_index = 0
        for j in range(i, len(matrix[i])):
            # See if two rows need to be changed
            if abs(matrix[j][i]) > max_factor:
                max_factor = abs(matrix[j][i])
                max_factor_index = j
        
        if max_factor_index != 0:
            # Swapping two rows
            storage = matrix[i].copy()
            #print('SW1\n',matrix, vector, '\n')
            matrix[i] = matrix[max_factor_index]
            #print('SW2\n',matrix, vector, '\n')
            matrix[max_factor_index] = storage
            #print('SW3\n',matrix, vector, '\n')

            storage = vector[i]
            vector[i] = vector[max_factor_index]
            vector[max_factor_index] = storage
            
        #print('SWP\n',matrix, vector, '\n')
        
        """ # This part can be uncommented if you want to have the pivots be equal to one.
        scaling_factor = (1 / matrix[i][i])
        matrix[i] = scaling_factor * matrix[i]
        vector[i] = scaling_factor * vector[i]
        
        print('SCL\n', matrix, vector, scaling_factor, '\n')"""
        
        for k in range(i + 1, len(matrix[i])):
            swap_factor = matrix[k][i] / matrix[i][i]
            matrix[k] = matrix[k] -  swap_factor * matrix[i] 
            vector[k] = vector[k] - swap_factor * vector[i]
    
        #print('SUB\n',matrix, vector, '\n')
        
    #print('RES\n', matrix, vector, '\n')
    return matrix, vector# Not needed but nice to do.


def solve_system_of_equations(matrix, vector):
    matrix, vector = Guassian_elimination(matrix, vector)
    
    x = 0 * vector
    
    for i in [len(vector) - 1 - i for i in range(0, len(vector))]:
        x[i] = (vector[i] - sum([x[j] * matrix[i][j] for j in range(0, len(matrix[i]))])) / matrix[i][i]
    
    return x

def get_inverse_of_matrix(matrix):
    """
    A function which calculates the inverse of a matrix using the determinant. Not as quick as using Jordan elimination but much easier to impliment. https://youtu.be/xZBbfLLfVV4 and https://youtu.be/ArcrdMkEmKo
    """
    if len(matrix) != len(matrix[0]):
        raise ValueError('Can only handle square matrices')
    
    size_matrix = len(matrix)
    determinant = calculate_determinant(matrix)
    cofactor_matrix = 0 * matrix.copy()
    
    for i in range(0, size_matrix):
        for j in range(0, size_matrix):
            
            sub_matrix = np.array([[matrix[y][x] for x in range(0, size_matrix) if x != j] for y in range(0, size_matrix)  if y != i])
            #print(sub_matrix, '\n', i, j, '\n\n')
            
            cofactor_matrix[i][j] = (-1)**(i + j) * calculate_determinant(sub_matrix)
    
    #print(cofactor_matrix, '\n')
    inverse = np.transpose(cofactor_matrix) * ( 1 / determinant)
    
    return inverse



