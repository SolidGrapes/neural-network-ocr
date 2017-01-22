import sys
import constants
import numpy as np
from neuralnet import NeuralNet
from getSamples import *

# Driver program for instantiating and training a Multi-Layer Perceptron for
# Optical Character Recognition

def trainNeuralNet(numHiddenLayers=constants.NUM_LAYERS, 
                   neuronsPerLayer=constants.HIDDEN_LAYER, 
                   actThres=constants.ACT_THRES, 
                   bias=constants.BIAS, 
                   learningRate=constants.LEARNING_RATE, 
                   numTrainSamples=constants.NUM_TRAIN):
    numNeuronLayers = [constants.SAMPLE_SIZE]
    for _ in xrange(numHiddenLayers):
        numNeuronLayers.append(neuronsPerLayer)
    numNeuronLayers.append(len(constants.CHARACTERS))
    neuralnet = NeuralNet(numNeuronLayers, actThres, bias)
    trainSamples = []
    for character in constants.CHARACTERS:
        print("Obtaining Samples for Character {}".format(character))
        trainSamples.append(getTrainingSamples(character, numTrainSamples))
    # Randomly permutaing the order of samples used for Training. 
    for i in xrange(numTrainSamples):
        perm = np.random.permutation(constants.CHARACTERS)
        for char in perm:
            sample = trainSamples[char][i]
            new_sample = np.array(map(lambda x: 0 if x == 255 else 1, sample.flatten()))
            trueImage = list(np.zeros(len(constants.CHARACTERS)))
            trueImage[char] = 1.
            neuralnet.backPropagationTrain(new_sample, trueImage, learningRate)
    return neuralnet

def testNeuralNet(neuralnet, numTestSamples=constants.NUM_TEST):
    totalCorrect = []
    for character in constants.CHARACTERS:
        print("Testing Samples for Character {}".format(character))
        currCorrect = 0
        testSamples = getTestSamples(character, numTestSamples)
        for sample in testSamples:
            new_sample = np.array(map(lambda x: 0 if x == 255 else 1, sample.flatten()))
            prediction = neuralnet.feedForwardOutput(new_sample, printRanking=False)
            if prediction == character:
                currCorrect += 1
        print("Number of Predictions Correct for Character {}: {}".format(character, currCorrect))
        print("Prediction Precision for Character {}: {}".format(character, float(currCorrect)/numTestSamples))
        totalCorrect.append(currCorrect)
    print("Total Correct Predictions: {}".format(float(sum(totalCorrect))))
    print("Total Precision of Predictions: {}".format(
          float(sum(totalCorrect))/(numTestSamples * len(constants.CHARACTERS))))

# Prediction for single image
def testNeuralNetSingle(neuralnet, filename , visualizeImage=False):
    sample = getSample(filename)
    if visualizeImage:
        print(sample)
    new_sample = np.array(map(lambda x: 0 if x == 255 else 1, sample.flatten()))
    prediction = neuralnet.feedForwardOutput(new_sample, printRanking=False)
    print("Prediction: {}".format(prediction))

def parseFlags(stringList, train):
    attrDict = {}
    flagsDict = constants.TRAIN_FLAGS if train else constants.TEST_FLAGS
    specialFlags = constants.SPECIAL_FLAGS
    try:
        for flag in flagsDict:
            if flag in stringList:
                attrDict[flagsDict[flag]] = \
                    int(stringList[stringList.index(flag) + 1])
    except:
        raise ValueError('Improper Syntax')
    return attrDict

def main():
    trainAttr = parseFlags(sys.argv[1:], train=True)
    testAttr = parseFlags(sys.argv[1:], train=False)
    neuralnet = trainNeuralNet(**trainAttr)

    for flag in constants.SPECIAL_FLAGS:
        stringList = sys.argv[1:]
        if flag in stringList:
            filename = stringList[stringList.index(flag) + 1]
            if flag =='-s':
                testNeuralNetSingle(neuralnet, filename)
            else:
                testNeuralNetSingle(neuralnet, filename, True)
            break
    else:
        testNeuralNet(neuralnet, **testAttr)

if __name__ == '__main__':
    main()