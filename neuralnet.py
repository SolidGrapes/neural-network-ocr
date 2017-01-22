import numpy as np
import constants
from scipy import stats
from neuron import *

class NeuralNet(object):
    def __init__(self, numNeuronLayers=constants.DEFAULT_LAYERS, 
                 actThres=constants.ACT_THRES, bias=constants.BIAS):
        ''' Constructor for new Hidden, Input, and Output Layers of the Neural Net

        Args: 
            numNeuronLayers (int list): Variable Number of Neurons in each Neuron 
                                        Layer. For example, neuronLayers[i] denotes 
                                        the number of neurons in Neuron Layer i+1.
                                        Must have at least input and output layer.
            actThres           (float): Activation Threshold of Sigmoid
            bias               (float): Bias of Activation Function
        '''
        
        neuronLayers = [[Neuron([], []) for _ in xrange(numNeuronLayers[0])]]
        for i in xrange(1, len(numNeuronLayers)):
            prevLayer = neuronLayers[-1]
            neuronLayers.append(
                [Neuron([np.random.randint(-10,10) for _ in xrange(len(prevLayer))], 
                        actThres=actThres, 
                        bias=bias) for _ in xrange(numNeuronLayers[i])])
        self.neuronLayers = neuronLayers

    def feedForwardOutput(self, inputImage, printRanking=False):
        '''
            Args:
                inputImage (float list): vectorized form of image in pixels
        '''
        inputLayer = self.neuronLayers[0]
        for i in xrange(len(inputImage)):
            inputLayer[i].setSig(inputImage[i])

        for i in xrange(1, len(self.neuronLayers)):
            prevLayer = self.neuronLayers[i-1]
            for j in xrange(len(self.neuronLayers[i])):
                neuron = self.neuronLayers[i][j]
                sigs = list(map(lambda neuron: neuron.getSig(), prevLayer))
                weights = neuron.getUpstreamSynapseLayer()
                sigmoidInput = np.dot(sigs, weights)/float(neuron.getActThres()) + \
                               neuron.getBias()
                               

                neuron.setSig(stats.logistic.cdf(sigmoidInput))
                neuron.setSigDeriv(stats.logistic.pdf(sigmoidInput))
        if printRanking:
            print([self.neuronLayers[-1][i].getSig() 
                   for i in xrange(len(constants.CHARACTERS))])
        return max(xrange(len(constants.CHARACTERS)), 
                   key=lambda i: self.neuronLayers[-1][i].getSig())

    # Assumes sigmoids are up-to-date
    def _errorPropagation(self, trueImage):
        outputLayer = self.neuronLayers[-1]
        for i in xrange(len(outputLayer)):
            delta = (trueImage[i] - outputLayer[i].getSig()) * \
                     outputLayer[i].getSigDeriv()
            outputLayer[i].setDelta(delta)

        for i in xrange(len(self.neuronLayers)-2, 0, -1):
            currLayer = self.neuronLayers[i]
            nextLayer = self.neuronLayers[i+1]
            for j in xrange(len(currLayer)):
                weights = list(map(lambda neuron: 
                                        neuron.getUpstreamSynapseLayer()[j], 
                                        nextLayer))
                deltas = list(map(lambda neuron: neuron.getDelta(), nextLayer))
                new_delta = currLayer[j].getSigDeriv() * np.dot(deltas, weights)
                currLayer[j].setDelta(new_delta)

    # Assumes sigmoids are up-to-date and error has propagated.
    def _weightUpdate(self, learningRate=constants.LEARNING_RATE):
        outputLayer = self.neuronLayers[-1]
        prevLayer = self.neuronLayers[-2]
        for i in xrange(len(self.neuronLayers)-1, 0, -1): 
            currLayer = self.neuronLayers[i]
            prevLayer = self.neuronLayers[i-1]
            gradientDescent = learningRate * np.array(
                                list(map(lambda neuron: neuron.getSig(), prevLayer)))
            for neuron in currLayer:
                newUpstreamSynapseLayer = np.array(neuron.getUpstreamSynapseLayer()) + \
                                         neuron.getDelta() * gradientDescent
                neuron.setUpstreamSynapseLayer(newUpstreamSynapseLayer) 

    def backPropagationTrain(self, inputImage, trueImage, 
                             learningRate=constants.LEARNING_RATE):
        self.feedForwardOutput(inputImage)
        self._errorPropagation(trueImage)
        self._weightUpdate(learningRate=learningRate)

    def getLayer(self, n):
        return self.neuronLayers[n]

    def printNeuralNetwork(self):
        for i in range(len(self.neuronLayers)):
            print(list(map(lambda neuron: neuron.getSig(), self.neuronLayers[i])))