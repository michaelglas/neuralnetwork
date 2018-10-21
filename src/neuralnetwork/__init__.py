
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
        stopped = []
        res = []
        for it in self.iter:
            try:
                res.append(next(it))
            except StopIteration:
                stopped.append(True)
            else:
                stopped.append(False)
        if True in stopped:
            if False in stopped:
                raise TypeError("help help")
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

def random_offset(it,min_offset):
    for i in it:
        yield i+random.randint(-10,10)*min_offset

def random_offset_nested(it,min_offset):
    ret = []
    for i in it:
        if isinstance(i, (int,float)):
            ret.append(i+random.randint(-10,10)*min_offset)
        else:
            ret.append(random_offset_nested(i, min_offset))
    return ret
        

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
    def train(self, inputs, outputs, its, gensize, offset, offset_drop, cost_func):
        nns = [layeredneuralnetwork.from_values(self.layers, self.act, random_offset_nested(self.bias, offset), random_offset_nested(self.weights, offset)) for i in range(gensize)]
        old_offset = offset
        for i in range(its):
            print("please let this computer run")
            INDEX = i
            nns_cost = []
            for nn in nns:
                nn_costs = []
                for inp,out in better_zip(inputs,outputs):
                    nn_costs.append(cost_func(nn.calculate(*inp),out))
                bisect.insort(nns_cost,(statistics.mean(nn_costs),nn))
            try:
                if nns_cost[0][0]<best_nn[0]:  # @UndefinedVariable
                    best_nn=nns_cost[0]
            except NameError:
                best_nn=nns_cost[0]
            nns = []
            for j in nns_cost[:len(nns_cost)//2]:
                nns.append(layeredneuralnetwork.from_values(self.layers, self.act, random_offset_nested(j[1].bias, offset), random_offset_nested(j[1].weights, offset)))
                nns.append(layeredneuralnetwork.from_values(self.layers, self.act, random_offset_nested(j[1].bias, offset), random_offset_nested(j[1].weights, offset)))
            if len(nns)!=gensize:
                nns.append(layeredneuralnetwork(*self.layers,act=self.act))
            offset = old_offset
            for j in range(i//offset_drop):
                offset /= 1.2
            print("iteration %s of %s successful"%(i,its))
        self.set_values(best_nn[1].bias,best_nn[1].weights)
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
        
            
                    
                    
        
    


            