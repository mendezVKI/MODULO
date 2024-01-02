import numpy as np


def compute_K(datafile):
    # this is to compute the matrix K given a numpy array saved in npz
    data = np.load(datafile)

    D = data['D']
    K = np.dot(D.transpose(), D)

    # Save as numpy array all the data
    np.savez('Correlation_K',K=K)