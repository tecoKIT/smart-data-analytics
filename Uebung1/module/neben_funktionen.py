import matplotlib.pyplot as plt
import numpy as np

def plot2D_points(X, y):
    '''
    The function returns a 2D-scatter plot of rejected (in blue) and admitted cases (in red)
    
    Parameters
    -----------
    X (np.array)    Numerical numpy array for x-axis
    y (np.array)    Numerical numpy array for y-axis
    
    Returns
    --------
    figure    Scatter plot, showing rejected cases in blue and admitted in red
    '''
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')