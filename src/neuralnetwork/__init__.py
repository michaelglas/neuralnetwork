
import random
import json
import math
import operator
from functools import reduce
import bisect
import statistics

global INDEX

class better_zip:
    def __init__(self,*iter):
        self.iter = iter
    def __next__(self):
        res = []
        stopped = False
        for it in self.iter:
            try:
                res.append(next(it))
            except StopIteration:
                stopped = True
        if stopped:
            raise StopIteration
        return tuple(res)
    def __iter__(self):
        return self

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

def _random_offset_nested(it,max_offset,mutation_rate):
    ret = []
    for i in it:
        if isinstance(i, (int,float)):
            if random.randrange(mutation_rate)<1:
                ret.append(i+random.random()-0.5*max_offset)
            else:
                ret.append(i)
        else:
            ret.append(random_offset_nested(i, max_offset, mutation_rate))
    return ret

def random_offset_nested(it,max_offset,mutation_rate):
    max_offset *= 2
    return _random_offset_nested(it, max_offset, mutation_rate)
        

class NeuralNetworkEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, neuralnetwork):
            if isinstance(o, layeredneuralnetwork):
                return {"layers":o.layers,"weights":o.weights,"bias":o.bias,"act":o.act.__name__}
            return {"neurons":o.neurons,"weights":o.weights,"bias":o.bias,"input":o.input,"output":o.output,"act":o.act.__name__}
            
        json.JSONEncoder.default(self, o)

def NeuralNetworkDecoder(d):
    k = d.keys()
    if "neurons" in k and "weights" in k and "bias" in k and "input" in k and "output" in k and "act" in k:
        ret = object.__init__(neuralnetwork)
        ret.neurons = d["neurons"]
        ret.weights = d["weights"]
        ret.bias = d["bias"]
        ret.input = d["input"]
        ret.output = d["output"]
        ret.act = eval(d["act"])
        return ret
    elif "layers" in k and "weights" in k and "bias" in k and "act" in k:
        ret = layeredneuralnetwork.__new__(layeredneuralnetwork)
        ret.layers = d["layers"]
        ret.weights = d["weights"]
        ret.bias = d["bias"]
        ret.act = eval(d["act"])
        return ret

class neuralnetwork:
    def __init__(self,neurons,input,output,act=sigmoid):
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
        self.bias = [[random.randrange(-i,i) for j in range(i)] for i in layers]
        self.weights = [[[random.randrange(-l,l) for j in range(l)] for y in range(layers[i+1])] for i,l in enumerate(layers[:-1])]
    def calculate(self, *input):  # @ReservedAssignment
        values = input
        # iterate over layers
        for i in enumerate(self.weights):
            new_values = []
            out = []
            biases = self.bias[i[0]]
            # iterate over input weights of neurons in layer i
            for j in enumerate(i[1]):
                value = 0
                # iterate over all inputs of an individual neuron
                # and scalar multiply input value with weight.
                for l in zip(j[1],values):
                    value+=l[0]*l[1]
                    
                value *= biases[j[0]]
                
                new_values.append(self.act(value))
                out.append(value)
            values = new_values
        return out
    @classmethod
    def from_values(cls,layers,act,bias,wheights):
        self = cls.__new__(cls)
        self.layers=layers
        self.act=act
        self.bias = bias
        self.weights = wheights
        return self
    def set_values(self,bias,wheights):
        self.bias = bias
        self.weights = wheights
    # genetic algorithm
    def train(self, inputs, outputs, its, gensize, offset, offset_drop, fitness_func, mutation_rate):
        nns = [layeredneuralnetwork.from_values(self.layers, self.act, random_offset_nested(self.bias, offset, mutation_rate), random_offset_nested(self.weights, offset, mutation_rate)) for i in range(gensize)]
        old_offset = offset
        train_len = 0
        for i in inputs:
            train_len+=1
        ol = 0
        for i in outputs:
            ol+=1
        if train_len!=ol:
            raise ValueError("inputs and outputs are not the same length")
        
        best_nn = 0
        best_fitness = 0
        for i in range(its):
            print("please let this computer run")
            cum_weights = []
            old_weight = 0
            for nn in nns:
                nn_fitnesses = 0
                for inp,out in better_zip(inputs,outputs):
                    nn_fitnesses += fitness_func(nn.calculate(*inp),out)
                fitness = nn_fitnesses/train_len
                cum_weights.append(old_weight+fitness)
                if fitness>best_fitness:
                    best_fitness = fitness
                    best_nn = nn
            total = cum_weights[-1]
            parents = (nns[bisect.bisect(cum_weights, random.random() * total, 0, gensize)] for j in range(gensize))
            nns = [layeredneuralnetwork.from_values(self.layers, self.act, random_offset_nested(parent.bias, offset, mutation_rate), random_offset_nested(parent.weights, offset, mutation_rate)) for parent in parents]
            offset = old_offset
            for j in range(i//offset_drop):
                offset /= 1.2
            print("iteration %s of %s successful"%(i,its))
        self.set_values(best_nn.bias,best_nn.weights)
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