def string_to_num(string, translation_tables, split_method):
    """
    Based on this stackoverflow thread: https://stackoverflow.com/questions/493174/is-there-a-way-to-convert-number-words-to-integers
    Takes written number (one, nine, twenty two, etc) as a string, translation tables which is a list with lists and a string for the split method. Those lists of the translation tabels are spelled out numbers (in order) with the first key of the list being a list whos first value is a float/int of the first string and whos second value is the steps between values of the strings. 
    example of translation table:
    ones=[[1,1], 'one', 'two', 'three', 'four', 'five','six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'nineteen']#from one to 
    tens=[[20,10], 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    translation_table=[ones, tens]
    """
    try:
        count=float(string)
        parts=[]
    except ValueError:
        if type(string)==str:
            count=0
            parts=string.lower().split(split_method)#makes it all lower characters and splits based on space
        else:
            print('Given argument is not a string and cannot be floated.')
    
    for part in parts:
        for table in translation_tables:
            start=table[0][0]
            step=table[0][1]
            for key in range(1,len(table)):
                value=(key-1)*step+start
                if table[key]==part:
                    count=count+value
    
    return count
