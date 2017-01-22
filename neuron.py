import numpy as np
import constants


class Neuron(object):
    def __init__(self, upstreamSynapseLayer, 
                 actThres=constants.ACT_THRES, bias=constants.BIAS): 
        '''Constructs a Neuron
        
        Args: 
            upstreamSynapseLayer   (Synapse list):
                Adjacent Synapse Layer, with current Neuron as heads of synapses
            actThres                      (float): 
                Activation Threshold of Sigmoid
            bias                          (float): 
                Bias of Activation Function
        '''
        self.upstreamSynapseLayer = upstreamSynapseLayer
        self.actThres= actThres
        self.bias = bias
        
    # Setters and Getters for instance attributes
    def setActThres(self, actThres):
        self.actThres= actThres

    def getActThres(self):
        return self.actThres

    def setBias(self, bias):
        self.bias = bias

    def getBias(self):
        return self.bias

    def setUpstreamSynapseLayer(self, synapseLayer):
        self.upstreamSynapseLayer = synapseLayer

    def getUpstreamSynapseLayer(self):
        return self.upstreamSynapseLayer
    
    def setSig(self, sig):
        self.sig = sig

    def getSig(self):
        try: 
            return self.sig
        except AttributeError:
            raise AttributeError("Sigmoid needs to be computed for this Neuron")

    def setSigDeriv(self, sigDeriv):
        self.sigDeriv = sigDeriv

    def getSigDeriv(self):
        try:
            return self.sigDeriv
        except AttributeError:
            raise AttributeError("Sigmoid needs to be computed for this Neuron")

    # Memoized Error Propagation Computations
    def setDelta(self, delta):
        self.delta = delta 

    def getDelta(self):
        try:
            return self.delta
        except:
            raise AttributeError("Error Delta needs to be computed for this Neuron")

