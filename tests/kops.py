#!/usr/bin/python3

import neuralnetwork
import re
import json
import os.path

DATADIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),"testdata/kops_155rflot_1984-1996")

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

def cost_func(out1,out2):
    res=0
    for i,j in zip(out1,out2):
        res += abs(i-j)
    return res

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

io = open(os.path.join(DATADIR,"nn.json"),"r")
nn = json.load(io,object_hook=neuralnetwork.NeuralNetworkDecoder)
io.close()
print(nn.calculate(*height[1000:1005:2]+temperature[1000:1005:2]),displacement[505:510],)
"""
nn.train(it2(it(height[:2000:2],5),it(temperature[:2000:2],5)),it(displacement[5:1005],5),1500,1001,0.5,100,cost_func)
"""

