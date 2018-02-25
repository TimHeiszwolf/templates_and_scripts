def min_key(dic):
    """
    Based on https://stackoverflow.com/questions/42044090/return-the-maximum-value-from-a-dictionary
    Returns the key of the min value in a dictionairy
    """
    return [k for k, v in dic.items() if v == min(dic.values())][0]