# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>
import sys
import math
import random
import string
import json as ujson
from operator import itemgetter as ig
from operator import sub
from math import e
# random.seed(1)
# random.seed(19)

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
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-.20, .20)
                
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)
        self.error = -99999

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[1], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=100, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
                # if error == self.error: return
                self.error = error          
            print('error %s | %-.5f' % (i, error))

    def accuracy(self, patterns):                        
        guessable, correction = zip(*patterns)
        guess = [self.update(g) for g in guessable]        
        correctionRanks = [map(ig(0), sorted(enumerate(c), key=ig(1), reverse=True)) for c in correction]
        guessRanks = [map(ig(0), sorted(enumerate(c), key=ig(1), reverse=True)) for c in guess]        
        for c, g in zip(correctionRanks, guessRanks):            
            print c[0], g

def normalize(func):
	def aply():
		from operator import itemgetter as ig
		import code
		from operator import sub
		results = list(func())
		_in = map(ig(0), results)
		out = map(ig(1), results)
		_in_ = zip(*_in)
		least = map(min, _in_)
		most = map(max, _in_)
		distance = map(sub, most, least)
		
		code.interact(local=locals())
		# return r
	return aply

@normalize
def getDataSet():
    dataSet = ujson.load(open('uNNTrain'))
    exprs = ["AN", "DI", "FE", "HA", "NE", "SA", "SU"]
    for elem in dataSet:        
        s = sum(elem["values"])
        values = [int((v * 10000)/float(s)) for v in elem["values"]]
        yield [values, [1.0 if elem['class'] == expr else -1.0 for expr in exprs]]        

def demo(I=11, O=1, sleep=0):    
	l = getDataSet()	

    # I = 112
    # O = 7
    # H = (I + O) / 2
    # pat = list(getDataSet())        
    # n = NN(I, H, O)    
    # n.train(pat, 10000, 0.02, 0.01)
    # print n.accuracy(pat)

if __name__ == '__main__':    
    demo(*map(int, sys.argv[1:]))
