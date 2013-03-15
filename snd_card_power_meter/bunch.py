from __future__ import print_function

class Bunch(object):
    def __init__(self, **kwds):
        # Adapted from 
        # http://stackoverflow.com/questions/35988/c-like-structures-in-python        
        self.__dict__.update(kwds)

    def __str__(self):
        # Adapted from Simon Brunning's code here:
        # http://code.activestate.com/recipes/52308/#c5
        state = ["{}={}".format(attribute, value)
                 for (attribute, value)
                 in self.__dict__.items()]
        return '\n'.join(state)

    def __eq__(self, other):
        # Equality code from http://stackoverflow.com/q/390250/732596
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)
