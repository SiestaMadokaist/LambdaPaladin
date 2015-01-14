# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>
import sys
import numpy as np
import math
import random
import string
import ujson
from operator import add, sub, mul, div, itemgetter as ig
from math import e
import code

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(i, j, fill=0.0):
    return [[fill] * j for _ in xrange(i)]    

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):            
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1 for _ in xrange(self.ni)]
        self.ah = [1 for _ in xrange(self.nh)]        
        self.ao = [1 for _ in xrange(self.no)]
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in xrange(self.ni):
            for j in xrange(self.nh):
                self.wi[i][j] = rand(-.20, .20)
                
        for j in xrange(self.nh):
            for k in xrange(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)
        self.error = -99999

    def update(self, inputs):
        assert len(inputs) == self.ni - 1, "wrong number of inputs"        

        # input activations
        for i in xrange(self.ni-1):            
            self.ai[i] = inputs[i]

        # hidden activations
        for j in xrange(self.nh):
            accumulator = 0.0
            for i in xrange(self.ni):
                accumulator = accumulator + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(accumulator)

        # output activations
        for k in xrange(self.no):
            accumulator = 0.0
            for j in xrange(self.nh):
                accumulator = accumulator + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(accumulator)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        assert len(targets) == self.no, "wrong number of target values"        

        def outDeltaGenerator():
            for target, activator in zip(targets, self.ao):                
                yield dsigmoid(activator) * (target - activator)

        def hiddenDeltaGenerator():
            for output_weights in self.wo:
                yield sum(odelta * weight for odelta, weight in zip(output_deltas, output_weights))    

        output_deltas = list(outDeltaGenerator())
        hidden_deltas = list(hiddenDeltaGenerator())

        # update output weights
        for j in xrange(self.nh):
            for k in xrange(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in xrange(self.ni):
            for j in xrange(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change
        
        error = sum(0.5 * (target - activator) ** 2 for target, activator in zip(targets, self.ao))
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[1], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in xrange(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in xrange(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=100, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]                
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
                # if error == self.error: return
                self.error = error          
            print('error %s | %-.5f' % (i, error))

    # def accuracy(self, patterns):                        
    #     guessable, correction = zip(*patterns)
    #     guess = [self.update(g) for g in guessable]        
    #     correctionRanks = [map(ig(0), sorted(enumerate(c), key=ig(1), reverse=True)) for c in correction]
    #     guessRanks = [map(ig(0), sorted(enumerate(c), key=ig(1), reverse=True)) for c in guess]        
    #     for c, g in zip(correctionRanks, guessRanks):            
    #         print c[0], g[0]


    def accuracy(self, patterns):                        
        guessable, correction = zip(*patterns)
        guess = [self.update(g) for g in guessable]        
        correctionRanks = [map(ig(0), sorted(enumerate(c), key=ig(1), reverse=True)) for c in correction]
        guessRanks = [map(ig(0), sorted(enumerate(c), key=ig(1), reverse=True)) for c in guess]        
        confusionMatrix = [[0 for i in xrange(7)] for i in xrange(7)]
        for c, g in zip(correctionRanks, guessRanks):
            confusionMatrix[c[0]][g[0]] += 1
        print sum(c[0] == g[0] for c, g in zip(correctionRanks, guessRanks)) / float(len(patterns))
        for mat in confusionMatrix:
            print mat


def normalized(self):
    def g1(ins, lowerbound, diffs):
        for elem in ins:
            yield  map(div, (map(sub, elem, lowerbound)), diffs)

    def function():
        dataset = list(self())
        ins = map(ig(0), dataset)
        outs = map(ig(1), dataset)        
        iins = zip(*ins)
        leasts = map(min, iins)
        mosts = map(max, iins)
        diffs = map(float, map(sub, mosts, leasts))
        delta = map(mul, mosts, [1/8.0] * len(mosts))
        upperbound = map(add, mosts, delta)
        lowerbound = map(sub, leasts, delta)
        xranges = map(sub, upperbound, lowerbound)        
        normalizeddata = list(g1(ins, lowerbound, xranges))        
        return map(list, zip(normalizeddata, outs))
    return function

def xorset():
    return [
        [[0, 0], [0]],
        [[1, 0], [1]],
        [[0, 1], [1]],
        [[1, 1], [0]]
    ]

@normalized
def getDataSet():
    dataset = ujson.load(open('uNNTrain'))
    exprs = ["AN", "DI", "FE", "HA", "NE", "SA", "SU"]
    for elem in dataset:
        s = sum(elem["values"])
        values = [int((v * 10000)/float(s)) for v in elem["values"]]        
        yield [values, [1.0 if elem['class'] == expr else -1.0 for expr in exprs]]

def randomSet():
    for i in xrange(100):
        ins = [random.randint(0, 1) for i in xrange(100)]
        outs = [random.randint(0, 1)]
        yield [ins, outs]

def load():
    nn = NN(112, 10, 7)
    nn.wi = []
    nn.wo = []    
    for i, line in enumerate(open('weights.txt', 'rb')):    
        weights = line.replace("[","").replace("]","")
        iweights = map(float, weights.split(","))
        func = nn.wi.append if len(iweights) == 7 else nn.wo.append
        func(iweights)
    return nn

def demo(h=10, lr=0.02, m=0.01):    
    H = int(h)
    learnrate = float(lr)
    momentum = float(m)
    n = NN(112, H, 7)    
    pat = getDataSet()
    random.shuffle(pat)
    train = pat
    validation = pat[85:]
    n.train(train, 1000, learnrate, momentum)
    n.accuracy(validation)
    code.interact(local=locals())

if __name__ == '__main__':    
    demo(*sys.argv[1:])