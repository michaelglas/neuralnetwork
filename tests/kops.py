#!/usr/bin/python3

import neuralnetwork
from neuralnetwork import sigmoid
import re
import json
import os.path
import numpy

DATADIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),"testdata/kops_155rflot_1984-1996")

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

class it:
    def __init__(self,l,len):
        self.l = l
        self.len = len
        self._index = 0
    def __next__(self):
        try:
            l = self.l[self._index:self._index+self.len]
            if len(l)<self.len:
                self._index = 0
                raise StopIteration
            self._index += 1
            return l
        except IndexError:
            self._index = 0
            raise StopIteration
    def __iter__(self):
        return self

class it2:
    def __init__(self,*iter):
        self.iter = iter
    def __next__(self):
        stopped = False
        try:
            res = next(self.iter[0])
        except StopIteration:
            stopped=True
            res = []
        for it in self.iter[1:]:
            try:
                res += next(it)
            except StopIteration:
                stopped=True
        if stopped:
            raise StopIteration
        return res
    def __iter__(self):
        return self

global train
global train_length

def fitness_func(nn):
    fitnesses = 0
    for inp,out in train:
        fitnesses += 1/abs((nn.calculate(inp)-out.T).sum())
    return fitnesses/train_length

regex = re.compile(",([\d\.]*)\n")
io = open(os.path.join(DATADIR,"kops_155rflot_1984-1996_displacement.csv"))
displacement = [float(i.group(1)) for i in regex.finditer(io.read())]
io.close()
io = open(os.path.join(DATADIR,"kops_155rflot_1984-1996_height.csv"))
height = [float(i.group(1)) for i in regex.finditer(io.read())]
io.close()
io = open(os.path.join(DATADIR,"kops_155rflot_1984-1996_temperature.csv"))
temperature = [float(i.group(1)) for i in regex.finditer(io.read())]
io.close()

to_mat = lambda y:map(lambda x: numpy.matrix(x).T, y)

train = better_zip(to_mat(it2(it(height[:1000:2],5),it(temperature[:1000:2],5))),to_mat(it(displacement[:500],5)))
train_length = 996

io = open(os.path.join(DATADIR,"nn.json"),"r")
nn = json.load(io,object_hook=neuralnetwork.NeuralNetworkDecoder)
io.close()

l = 0

nn.train(1,100,1,1000,fitness_func,1)

io = open(os.path.join(DATADIR,"nn.json"),"w")
nn = json.dump(nn,io,cls=neuralnetwork.NeuralNetworkEncoder)
io.close()


