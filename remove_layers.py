def remove_layers(data):
    """
    Removes all unneeded layers of dictionaries, lists and tuples
    """
    while True:
        if type(data)==int or type(data)==float or len(data)!=1:
            break
        
        if type(data)==dict:
            data=data[list(data.keys())[0]]
        else:
            data=data[0]
    
    return data