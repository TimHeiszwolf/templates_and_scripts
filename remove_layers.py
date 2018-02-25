def remove_layers(data):
    """
    Removes all unneeded layers of dictionaries, lists and tuples
    """
    while True:
        isdictlisttuple=type(data)==dict or type(data)==list or type(data)==tuple
        
        if type(data)==int or type(data)==float:
            break
        elif isdictlisttuple:
            if len(data)!=1:
                break
        
        if type(data)==dict:
            data=data[list(data.keys())[0]]
        elif isdictlisttuple and type(data)!=dict:
            data=data[0]
        else:
            break
    
    return data