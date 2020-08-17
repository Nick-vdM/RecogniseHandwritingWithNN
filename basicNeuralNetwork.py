# TODO: Make a menu of options to choose from (train, rate,

import math
import time
import random
import sys
import numpy as np
import gzip


def sigmoid(var):
    return np.divide(1.0, (np.add(1.0, np.exp(np.negative(var)))))


def pderiveOutToNet(var):
    return np.multiply(sigmoid(var), (np.subtract(1, sigmoid(var))))


def pderiveEtotalToOut(output, target):
    return np.subtract(output, target)


class neuralNetwork:
    def __init__(self, numberOfInputs, numberOfHiddens, numberOfOutputs,
                 trainDigitsX, trainDigitsY, testDigitX, testDigitY):
        """
        :param numberOfInputs:
        :param numberOfHiddens:
        :param numberOfOutputs:
        :param trainDigitsX:
        :param trainDigitsY:
        :param testDigitX:
        :param testDigitY:
        """
        self.layerSizes = [numberOfInputs, numberOfHiddens, numberOfOutputs]
        self.trainDigitsX = trainDigitsX
        self.trainDigitsY = trainDigitsY
        self.testDigitsX = testDigitX
        self.testDigitsY = testDigitY

        self.epochs = 1
        self.miniBatchSize = 20
        self.learningRate = 3

        # make these adaptable to a larger number of layers if wanted
        self.biases = np.array([np.random.rand(i, 1) for i in self.layerSizes[
                                                              1:]])
        self.weights = np.array([np.random.randn(i, j) for i, j in
                                 zip(self.layerSizes[1:], self.layerSizes[:-1])])

    def getOutput(self, X):
        """
        :param X: Input dataset
        :return: Output of the neural network
        """
        for bias, weight in zip(self.biases, self.weights):
            X = sigmoid(np.add(np.dot(weight, X), bias))
        return X

    def forwardFeed(self, X):
        """
        We save the nets and outputs for use in backpropogation
        :param X: Input to the neural network
        :return: nets, outputs
        """
        nets = []
        outputs = [X]  # the first layer's output is the input
        for bias, weight in zip(self.biases, self.weights):
            # The net value is actually just the dot product formula
            # so we can use this to speed things up a bit
            net = np.add(np.dot(weight, X), bias)
            nets.append(net)
            outputs.append(sigmoid(net))
        return nets, outputs

    def stochasticGradientDescent(self):
        trainSet = np.array(list(zip(self.trainDigitsX, self.trainDigitsY)))
        for i in range(self.epochs):
            print("Epoch", i, "started!")
            start = time.perf_counter()
            # we want to shuffle the dataset to keep the random order property
            random.shuffle(trainSet)
            batches = [
                trainSet[x:np.add(x, self.miniBatchSize)]
                for x in range(0, len(trainSet), self.miniBatchSize)
            ]
            print("aklsduhrflkjasdhfkljdsaf", len(batches))
            iteration = 0
            for batch in batches:
                print(iteration, len(batch))
                iteration += 1
                self.backwardsMiniBatch(batch)
            timeTaken = time.perf_counter() - start
            print("Took", timeTaken, "seconds to finish epoch", i)

    def backwardsMiniBatch(self, batch):
        # nabla = learning rate; these are also the partial derivative of error
        # in respect to the weight
        nablaBiases = [np.zeros(b.shape) for b in self.biases]
        nablaWeights = [np.zeros(w.shape) for w in self.weights]
        for input, output in batch:
            # we process through the nabla weights here
            deltaNablaBias, deltaNablaWeight = self.backPropogate(input, output)
            nablaBiases = [np.add(a, b) for a, b in zip(nablaBiases, deltaNablaBias)]
            nablaWeights = [np.add(a, b) for a, b in
                            zip(nablaWeights, deltaNablaWeight)]
        # actually update them by doing current weight - (learning rate per
        # batch) * nabla weight in each batch
        self.weights = [
            weight - (self.learningRate / self.miniBatchSize) * nablaW
            for weight, nablaW in zip(self.weights, nablaWeights)
        ]
        self.biases = [
            biases - (self.learningRate / self.miniBatchSize) * nablaB
            for biases, nablaB in zip(self.biases, nablaBiases)
        ]

    def backPropogate(self, input, expectedOutput):
        """
        :param input
        :param expectedOutput
        :return: tuple representing the nablaWeight and nablaBiases - the
        updated weights for this layer
        """
        nablaBiases = [np.zeros(b.shape) for b in self.biases]
        nablaWeights = [np.zeros(w.shape) for w in self.weights]

        nets, outputs = self.forwardFeed(input)
        # Do the first backwards pass and initialise our nablas; the first
        # pass is different because we use the expected outputs

        # A section that seems to have been skipped in the example is
        # calculating biases. In these, the value of partialNet / weight = 1
        # so we just don't multiply it by the previous layer's outputs
        delta = pderiveEtotalToOut(outputs[-1], expectedOutput) * \
                pderiveOutToNet(nets[-1])
        nablaBiases[-1] = delta
        # now we multiply by the previous layer's outs
        nablaWeights[-1] = np.dot(delta, outputs[-2].transpose())

        # now instead of doing a cost derivative, we can just use the
        # previously calculated delta values
        delta = np.multiply(np.dot(self.weights[-1].transpose(), delta),
                            pderiveOutToNet(nets[-2]))

        nablaBiases[-2] = delta
        trans = outputs[-3].transpose()
        nablaWeights[-2] = np.dot(delta, trans)
        return nablaBiases, nablaWeights

    def rate(self, testData):
        """
        Rates how well the neural network did
        :return:
        """
        results = [(np.argmax(self.forwardFeed(x)), y)
                   for x, y in testData]
        return sum(int(x == y) for (x, y) in results)

    def saveModel(self, location):
        # TODO: Make it possible to save states
        """
        Saves the model in the specified location
        :param location:
        :return:
        """
        pass

    def saveTest(self):
        pass


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


def writeIntoCSV(fileName, array):
    pass


if __name__ == '__main__':
    numberOfInputs = 784
    numberOfHiddens = 30
    numberOfOutputs = 10
    # The format of this data is a bit unique because we want to use dot
    # products to speed up calculations. To stop numpy from wanting to do
    # full matrix maths and expanding it into a square (LxL) dimensions and
    # keeping it Lx1, we need to make a normal array e.g. [5, 6, 7] into [[
    # 5], [6], [7]] which we can do with reshape inside of these functions
    trainDigitsX = getXValuesFromCSV("data/TrainDigitX.csv.gz", numberOfInputs)
    print("Loaded train digit X")
    trainDigitY = getYValuesFromCSV("data/TrainDigitY.csv.gz", numberOfOutputs)
    print("Loaded train digit Y")
    testDigitX = getXValuesFromCSV("data/TestDigitX.csv.gz", numberOfInputs)
    print("Loaded test digit X")
    predictDigitY = getYValuesFromCSV("data/TestDigitY.csv.gz", numberOfOutputs)
    print("Loaded test digit Y")
    if len(sys.argv) < 8:
        print("""
        No commandline arguments passed; Reverting to default values:
            numberOfInputs = 784
            numberOfHiddens = 30
            numberOfOutputs = 10
            trainDigitsX = 'data/TrainDigitX.csv.gz'
            trainDigitY = 'data/TrainDigitY.csv.gz'
            testDigitX = 'data/TestDigitX.csv.gz'
            predictDigitY = 'data/TestDigitY.csv.gz'
        """)
    else:
        numberOfInputs = sys.argv[1]
        numberOfHiddens = sys.argv[2]
        numberOfOutputs = sys.argv[3]
        trainDigitsX = getXValuesFromCSV(sys.argv[4], numberOfInputs)
        trainDigitY = getYValuesFromCSV(sys.argv[5], numberOfOutputs)
        testDigitX = getXValuesFromCSV(sys.argv[6], numberOfInputs)
        predictDigitY = getYValuesFromCSV(sys.argv[7], numberOfOutputs)

    print("one!")
    NN = neuralNetwork(numberOfInputs, numberOfInputs, numberOfOutputs,
                       trainDigitsX, trainDigitY, testDigitX, predictDigitY)
    print("two!")
    NN.stochasticGradientDescent()
    print("done!")
