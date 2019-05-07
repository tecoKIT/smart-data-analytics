import numpy as np


def is_one_hot_encoded(a):
    """Checks if the rows in the array or dataframe `a` are one-hot encoded."""
    if np.any(np.logical_and(a != 1, a != 0)):
        # Return False if other values than 1 or 0 are present.
        return False
    if np.any(a.sum(axis=1) != 1):
        # Return False if any row contains more than one 1.
        return False
    # Else return True
    return True


def one_hot_to_ordinal(a):
    """Converts one-hot encoded array or dataframe to a 1D-array with ordinal numbers."""
    def index_of_one(a):
        # Returns number of column that contains the 1
        return np.where(a == 1)[0]

    # Reduce columns of a to one column by applying the index_of_one function.
    return np.apply_along_axis(func1d=index_of_one, axis=1, arr=a)
