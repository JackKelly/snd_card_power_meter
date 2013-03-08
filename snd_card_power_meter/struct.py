class Struct(object):
    # Adapted from 
    # http://stackoverflow.com/questions/35988/c-like-structures-in-python
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
