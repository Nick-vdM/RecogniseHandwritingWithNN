import math
import random
import sys
import numpy as np


def sigmoid(var):
    return 1 / 1 + math.exp(-var)


def sigmoidPrime(var):
    return sigmoid(var) * (1 - sigmoid(var))


class neuralNetwork:
    def __init__(self):
        self.layerSizes = [2, 2, 2]
        self.trainDigitsX = [[0.1, 0.1], [0.1, 0.2]]
        self.trainDigitsY = [[1, 0], [0, 1]]

        self.epochs = 1
        self.miniBatcheSize = 2
        self.learningRate = 0.1

        # make these adaptable to a larger number of layers if wanted
        self.biases = [np.random.rand(i, 1) for i in self.layerSizes[1:]]
        self.biases = [
            np.array([[0.1], [0.1]]),
            np.array([0.1], [0.1])
        ]
        self.weights = [
            np.array([[0.1, 0.2], [0.1, 0.1]]),
            np.array([[0.1, 0.1], [0.1, 0.2]]),
        ]

    def forwards(self, X):
        iteration = 0
        for bias, weight in zip(self.biases, self.weights):
            print("iteration: ", iteration)

            net = np.dot(weight, X) + bias
            print("net", iteration, " =", net)
            X = sigmoid(net)
            print("out", iteration, " =", X)
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

    def backwardsMiniBatch(self, batch):
        nablaBiases = [np.zeros(b.shape) for b in self.biases]
        nablaWeights = [np.zeros(w.shape) for w in self.weights]
        for i, j in batch:
            deltaNablaBias, deltaNablaWeight = self.backPropogate(i, j)
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
        nablaBiases = [np.zeros(b.shape) for b in self.biases]
        nablaWeights = [np.zeros(b.shape) for b in self.weights]

        activate = i, activates = [i]
        zValues = []

        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activate) + bias
            zValues.append(z)
            activate = sigmoid(z)
            activates.append(activate)
        # go back
        delta = self.pDerivative(activates[-1], j) * sigmoidPrime(zValues[-1])
        nablaBiases[-1] = delta
        nablaWeights[-1] = np.dot(delta, activates[-2].transpose())

        for k in range(2, 3):
            z = zValues[-1]
            primeSigmoid = sigmoidPrime(z)
            delta = np.dot(self.weights[-k - 1].transpose(),
                           delta) * primeSigmoid
            nablaBiases[-1] = delta
            nablaWeights[-1] = np.dot(delta, activates[-k - 1].transpose())
        return nablaBiases, nablaWeights

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


if __name__ == '__main__':
    NN = neuralNetwork()
