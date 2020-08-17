import math
import random
import sys
import numpy as np


def sigmoid(var):
    return 1.0 / (1.0 + np.exp(-var))


def pderiveOutToNet(var):
    return sigmoid(var) * (1 - sigmoid(var))

def pderiveEtotalToOut(output, target):
    return output - target

class threeLayerNeuralNetwork:
    """
    This neural network only allows three layers at the most. This just sped
    up implementation time and made a few finicky things easier as it wasn't
    needed
    """
    def __init__(self):
        self.layerSizes = [2, 2, 2]
        self.trainDigitsX = np.array([[[0.05], [0.10]]])
        self.trainDigitsY = np.array([[[0.01], [0.99]]])

        self.epochs = 1
        self.miniBatchSize = 1
        self.learningRate = 0.1

        self.biases = [
            np.array([[0.35], [0.35]]),
            np.array([[0.60], [0.60]])
        ]
        self.weights = [
            np.array([[0.15, 0.20], [0.25, 0.30]]),
            np.array([[0.40, 0.45], [0.50, 0.55]])
        ]

    def getOutput(self, X):
        """
        :param X: Input dataset
        :return: Output of the neural network
        """
        for bias, weight in zip(self.biases, self.weights):
            X = sigmoid(np.dot(weight, X) + bias)
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
            net = np.dot(weight, X) + bias
            nets.append(net)
            outputs.append(sigmoid(net))
        return nets, outputs

    def stochasticGradientDescent(self):
        trainSet = np.array(list(zip(self.trainDigitsX, self.trainDigitsY)))
        for i in range(self.epochs):
            # we want to shuffle the dataset to keep the random order property
            random.shuffle(trainSet)
            batches = [
                trainSet[x:x + self.miniBatchSize]
                for x in range(0, len(trainSet), self.miniBatchSize)
            ]
            for batch in batches:
                self.backwardsMiniBatch(batch)

    def backwardsMiniBatch(self, batch):
        # nabla = learning rate; these are also the partial derivative of error
        # in respect to the weight
        nablaBiases = [np.zeros(b.shape) for b in self.biases]
        nablaWeights = [np.zeros(w.shape) for w in self.weights]
        iterations = 0
        for input, output in batch:
            # we process through the nabla weights here
            iterations += 1
            deltaNablaBias, deltaNablaWeight = self.backPropogate(input, output)
            nablaBiases = [a + b for a, b in zip(nablaBiases, deltaNablaBias)]
            nablaWeights = [a + b for a, b in
                            zip(nablaWeights, deltaNablaWeight)]
            print("nabla weights #", str(iterations) + ":", nablaWeights)
            print("nabla biases #", str(iterations) + ":", nablaBiases)
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
        print("OVERALL WEIGHTS:", self.weights)
        print("OVERALL BIASES:", self.biases)

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
        delta = np.dot(self.weights[-1].transpose(), delta) * \
                pderiveOutToNet(nets[-2])

        nablaBiases[-2] = delta
        trans = outputs[-3].transpose()
        nablaWeights[-2] = np.dot(delta, trans)
        return (nablaBiases, nablaWeights)

    @staticmethod

    def rate(self, testData):
        """
        Rates how well the neural network did
        :return:
        """
        results = [(np.argmax(self.forwardFeed(x)), y)
                   for x, y in testData]
        return sum(int(x == y) for (x, y) in results)


if __name__ == '__main__':
    NN = threeLayerNeuralNetwork()
    print("First output:", end="")
    print(NN.getOutput(np.array([[0.1], [0.1]])))
    NN.stochasticGradientDescent()
    print("Second output:", end="")
    print(NN.getOutput(np.array([[0.1], [0.1]])))
