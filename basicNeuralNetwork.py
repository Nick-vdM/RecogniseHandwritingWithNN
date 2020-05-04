import math
import random
import sys
import numpy as np


def sigmoid(var):
    return 1 / 1 + math.exp(-var)


class neuralNetwork:
    def __init__(self, numberOfInputs, numberOfHiddens, numberOfOutputs,
                 trainDigitsX, trainDigitsY, testDigitX, predictDigitY):
        self.layerSizes = [numberOfInputs, numberOfHiddens, numberOfOutputs]
        self.trainDigitsX = trainDigitsX
        self.trainDigitsY = trainDigitsY
        self.testDigitsY = testDigitX

        self.epochs = 30
        self.miniBatcheSize = 20
        self.learningRate = 3

        # make these adaptable to a larger number of layers if wanted
        self.biases = [np.random.rand(i, 1) for i in self.layerSizes[1:]]
        self.weights = [np.random.randn(i, j)
                        for i, j in zip(self.layerSizes[:-1], self.layerSizes[1:])]

    def forwards(self, X):
        for bias, weight in zip(self.biases, self.weights):
            X = sigmoid(np.dot(weight, X) + bias)
        return X

    def stochasticGradientDescent(self):
        trainSet = zip(self.trainDigitsX, self.trainDigitsY)
        for i in range(self.epochs):
            random.shuffle(trainSet)
            batches = [
                trainSet[x:x + self.miniBatcheSize]
                for x in range(0, len(trainSet), self.miniBatcheSize)
            ]
            for batch in batches:
                self.backwardsMiniBatch(batch)
            pass

    def backwardsMiniBatch(self, batch):
        nablaBiases = [np.zeros(b.shape) for b in self.biases]
        nablaWeights = [np.zeros(w.shape) for w in self.weights]
        for i, j in batch:
            deltaNablaBias, deltaNablaWeight = self.backprop(i, j)
            nablaBiases = [a + b for a, b in zip(nablaBiases, deltaNablaBias)]
            nablaWeights = [a + b for a, b in
                            zip(nablaWeights, deltaNablaWeight)]
        self.weights = [
            weight - (self.learningRate / self.miniBatcheSize) * nablaW
            for weight, nablaW in zip(self.weights, nablaWeights)
        ]
        self.biases = [
            biases - (self.learningRate / self.miniBatcheSize) * nablaB
            for biases, nablaB in zip(self.biases, nablaBiases)
        ]

    def backPropogate(self, i, j):
        """
        :param i: positional in the network
        :param j: positional in the network
        :return: tuple representing the nablaWeight and nablaBiases
        """
        activate = i, activates = [i]
        zValues = []

        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activate) + bias
            zValues.append(z)
            activate = sigmoid(z)
            activates.append(activate)
        # go back
        delta = self.pDerivative

    @staticmethod
    def pDerivative(activationOutput, y):
        return activationOutput - y

    def rate(self, testData):
        """
        Rates how well the neural network did
        :return:
        """
        results = [(np.argmax(self.forwards(x)), y)
                   for x, y in testData]
        return sum(int(x == y) for (x, y) in results)


def getValuesFromCSV(fileName):
    filePath = "data/" + fileName
    file = np.loadtxt(fileName, dtype=np.double)


def writeIntoCSV(fileName, array):
    pass


if __name__ == '__main__':
    if len(sys.argv) < 8:
        print("ERROR: not enough commandline parameters passed",
              file=sys.stderr)
        exit(1)

    numberOfInputs = sys.argv[1]
    numberOfHiddens = sys.argv[2]
    numberOfOutputs = sys.argv[3]
    trainDigitsX = getValuesFromCSV(sys.argv[4])
    trainDigitY = getValuesFromCSV(sys.argv[5])
    testDigitX = getValuesFromCSV(sys.argv[6])
    predictDigitY = getValuesFromCSV(sys.argv[7])

    NN = neuralNetwork(numberOfInputs, numberOfInputs, numberOfOutputs,
                       trainDigitsX, trainDigitY, testDigitX, predictDigitY)
