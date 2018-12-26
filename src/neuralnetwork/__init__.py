
import random
import json
import math
import operator
from functools import reduce
import bisect
import statistics
import numpy

"""
def sigmoid(x):
    if x >= 0:
        z = numpy.exp(-x)
        return 1 / (1 + z)
    else:
        z = numpy.exp(x)
        return z / (1 + z)
"""
def sigmoid(x):
    s = 1/(1+numpy.exp(-x))
    return s

ACT_FUNCS = {"sigmoid":sigmoid}

def _random_offset_matrix(it,max_offset,mutation_rate):
    ret = numpy.empty_like(it)
    for i in range(it.size):
        if random.randrange(mutation_rate)<1:
            ret.put(i,it.item(i)+(random.random()-0.5)*max_offset)
        else:
            ret.put(i,it.item(i))
    """
    ret = []
    for i in it:
        if isinstance(i, (int,float)):
            if random.randrange(mutation_rate)<1:
                ret.append(i+random.random()-0.5*max_offset)
            else:
                ret.append(i)
        else:
            ret.append(random_offset_nested(i, max_offset, mutation_rate))
    """
    return ret

def random_offset_nested(it,max_offset,mutation_rate):
    max_offset *= 2
    ret = []
    for i in it:
        ret.append(_random_offset_matrix(i, max_offset, mutation_rate))
    return ret
        

class NeuralNetworkEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, numpy.matrixlib.defmatrix.matrix):
            return o.tolist()
        if isinstance(o, neuralnetwork):
            if isinstance(o, layeredneuralnetwork):
                return {"layers":o.layers,"weights":o.weights,"act":o.act.__name__,"fitness":o.fitness}
            return {"neurons":o.neurons,"weights":o.weights,"bias":o.bias,"input":o.input,"output":o.output,"act":o.act.__name__}
            
        json.JSONEncoder.default(self, o)

def NeuralNetworkDecoder(d):
    k = d.keys()
    print("test")
    if "neurons" in k and "weights" in k and "bias" in k and "input" in k and "output" in k and "act" in k:
        ret = object.__init__(neuralnetwork)
        ret.neurons = d["neurons"]
        ret.weights = d["weights"]
        ret.bias = d["bias"]
        ret.input = d["input"]
        ret.output = d["output"]
        ret.act = eval(d["act"])
        return ret
    elif "layers" in k and "weights" in k and "act" in k and "fitness" in k:
        ret = layeredneuralnetwork.__new__(layeredneuralnetwork)
        ret.layers = d["layers"]
        ret.weights = [numpy.matrix(i) for i in d["weights"]]
        ret.act = ACT_FUNCS[d["act"]]
        ret.fitness = d["fitness"]
        return ret

class neuralnetwork:
    def __init__(self,neurons,input,output,act=sigmoid):
        raise NotImplementedError()
        self.input = input
        self.output = output
        self.neurons = []
        self.weights = []
        self.bias = []
        self.act=act
        for i in neurons:
            self.neurons.append((zip(list(range(len(self.weights),len(self.weights)+len(i))),i),len(self.bias)))
            self.bias.append(random.randrange(-len(i),len(i),0.5))
            for j in i:
                self.weights.append(random.randrange(-len(i),len(i),0.5))
        self.fitness = 0
    def calculate(self,*input):
        pass
    def train(self,inputs,outputs,alg,its):
        pass

class layeredneuralnetwork(neuralnetwork):
    def __init__(self,*layers,act=sigmoid):
        if len(layers)<2:
            raise ValueError
        self.layers=layers
        self.act=act
        #self.bias = [[random.randrange(-i,i) for j in range(i)] for i in layers]
        self.weights = [numpy.matrix(numpy.random.rand(layers[i+1],l)) for i,l in enumerate(layers[:-1])]
        self.fitness = 0
    def calculate(self, input):  # @ReservedAssignment
        values = input
        # iterate over layers
        for i in self.weights[:-1]:
            values = self.act(i*values)
        return self.weights[-1]*values
    @classmethod
    def from_values(cls,layers,act,wheights):
        self = cls.__new__(cls)
        self.layers=layers
        self.act=act
        #self.bias = bias
        self.weights = wheights
        return self
    def set_values(self,wheights,fitness):
        #self.bias = bias
        self.weights = wheights
        self.fitness = fitness
    # genetic algorithm
    def train(self, its, gensize, offset, offset_drop, fitness_func, mutation_rate):
        nns = [layeredneuralnetwork.from_values(self.layers, self.act, random_offset_nested(self.weights, offset, mutation_rate)) for i in range(gensize)]
        old_offset = offset
        """
        train_len = 0
        for i in inputs:
            train_len+=1
        ol = 0
        for i in outputs:
            ol+=1
        if train_len!=ol:
            raise ValueError("inputs and outputs are not the same length")
        """
        
        best_nn = self
        best_fitness = self.fitness
        for i in range(its):
            print("please let this computer run")
            cum_weights = []
            weight = 0
            for nn in nns:
                fitness = fitness_func(nn)
                weight = weight+fitness
                cum_weights.append(weight)
                if fitness>best_fitness:
                    best_fitness = fitness
                    best_nn = nn
            total = cum_weights[-1]
            parents = (nns[bisect.bisect(cum_weights, random.random() * total, 0, gensize-1)] for j in range(gensize))
            nns = [layeredneuralnetwork.from_values(self.layers, self.act, random_offset_nested(parent.weights, offset, mutation_rate)) for parent in parents]
            offset = old_offset
            for j in range(i//offset_drop):
                offset /= 1.2
            print("iteration %s of %s successful"%(i,its))
            print("best fitness is %.6f"%best_fitness)
        self.set_values(best_nn.weights,best_fitness)
    def __lt__(self, b):
        return False
    def __le__(self, b):
        return True
    def __eq__(self, b):
        return True
    def __ne__(self, b):
        return False
    def __ge__(self, b):
        return True
    def __gt__(self, b):
        return False            