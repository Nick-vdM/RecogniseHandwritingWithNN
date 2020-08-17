#warning: this code is in my assignment. Don't directly copy it or you'll die
import numpy as np
import gzip

def getXValuesFromCSV(fileName, shape):
    unshaped_array = np.loadtxt(gzip.open(fileName, "rb"), delimiter=",")
    return np.array([
        np.reshape(i, (shape, 1)) for i in unshaped_array
    ])


def convertIntoNNOutputFormat(val, shape):
    """
    Turns the Y digit into the expected format. e.g.:
    0 = np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    1 = np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]) etc
    :param val:
    :return:
    """
    arr = np.zeros((shape, 1))
    arr[int(val)][0] = 1.0
    return arr


def getYValuesFromCSV(fileName, shape):
    # Get an array like [5, 3, 1, 6, 7, 1, 6 ... ]
    initialArray = np.loadtxt(gzip.open(fileName, "rb"), delimiter=",")
    # Change it into an array like np.array([[[0], [0], [0] etc that
    # the neural network understands
    return np.array([
        convertIntoNNOutputFormat(i, shape) for i in initialArray
    ])
