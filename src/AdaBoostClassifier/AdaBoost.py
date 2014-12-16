from __future__ import division
from Paladin import Paladin
import numpy as np
import sys
import random
import json
import math
def getSeeds(n):
    return [random.randint(-10, 10) for _ in xrange(n)]

class AdaBoost(object):
    """
    Adaboost implementation taken from https://gist.github.com/tristanwietsma/5486024 with a few change.
    """
    @classmethod            
    def loadSamples(cls, valid, invalid):
        for line in open(valid, 'rb'):
            yield (map(int, line.strip().split()), 1)

        for line in open(invalid, 'rb'):
            yield (map(int, line.strip().split()), -1)

    @classmethod
    def new(cls, shuffledSamples, trainPercentage, level):
        l = int(len(shuffledSamples) * trainPercentage / 100)
        trainSet = shuffledSamples[:l]
        validationSet = shuffledSamples[l:]
        ADA =  AdaBoost(trainSet, validationSet)
        seeds = [getSeeds(10) for _ in xrange(level)]
        consts = [random.randint(0, 1000) * 100 for _ in xrange(level)]
        for seed, const in zip(seeds, consts):
            paladin = Paladin(4, seed, const)
            ADA.add_paladin(paladin)
        return ADA
    
    def __init__(self, trainSet=None, validationSet=None, alphas=None, paladins=None):
        self.trainingSet = trainSet
        self.validationSet = validationSet
        self.n = len(self.trainingSet) if trainSet else 1
        self.weights = np.ones(self.n)/self.n
        self.alphas = [] if not alphas else alphas
        self.paladins = [] if not paladins else paladins

    @property
    def accuracy(self):
        targetSet, expectationSet = zip(*self.validationSet)
        return sum(e for e in self.evaluate(targetSet, expectationSet)) / len(targetSet)

    @property    
    def confusion(self):
        targetSet, expectationSet = zip(*self.validationSet)
        memo = [[0, 0], [0, 0]]
        for args, e in zip(targetSet, expectationSet):
            r = self.apply(*args)            
            memo[e == 1][r == 1] += 1
        return memo

    def updated_weights(self, errors, weights, alpha):
        for error, weight in zip(errors, weights):
            exponetor = alpha if error else -alpha
            yield weight * np.exp(exponetor)

    def add_paladin(self, paladin):
        errors = np.array([expectation != paladin.rule(*args) for args, expectation in self.trainingSet])
        et = (self.weights * errors).sum()
        alpha = 0.5 * np.log((1 - et) / et)
        w = np.array([weight for weight in self.updated_weights(errors, self.weights, alpha)])
        self.weights = w / w.sum()        
        self.alphas.append(alpha)
        self.paladins.append(paladin)

    def apply(self, *args):        
        hx = [alpha * paladin.rule(*args) for alpha, paladin in zip(self.alphas, self.paladins)]
        r = np.sign(sum(hx))
        return r

    def evaluate(self, targetSet, expectationSet):
        for args, expect in zip(targetSet, expectationSet):
            yield self.apply(*args) == expect

    def save(self, fout):         
        json.dump([{
            "Paladin": {
                "origin" : paladin.origin,
                "const": paladin.const,
                "N": paladin.N,
                },
            "Alpha": alpha
        } for paladin, alpha in zip(self.paladins, self.alphas)], open(fout, 'wb'))

def levelGenerator(n):            
    for i in xrange(1, n):
        lim = int(1.5 * i)
        repeat = 6 - math.log(lim)
        for e in xrange(int(repeat)):
            yield int(2.12 ** lim) * 12

def expand(iterablepairs):
    for x, rep in iterablepairs:
        for _ in xrange(rep):
            yield x

def main():    
    zexpand = [(10, 5), (20, 7), (30, 13), (35, 10), (50, 5), (100, 2), (200, 2), (250, 2), (350, 1), (500, 1), (1000, 1), (2000, 1)]
    levels = [e for e in expand(zexpand)]    
    # levels = [10, 10, 20, 20, 20, 20, 20, 30, 30, 35, 35, 35, 35, 50, 100, 100, 100, 200, 200, 200, 250, 500, 1000, 1000, 2000]        
    dataSet = [(x, l) for x, l in AdaBoost.loadSamples('../../.tmp/true-eye', '../../.tmp/false-eye')]    
    random.shuffle(dataSet)        
    for i, level in enumerate(levels):
        print level
        classifiers = AdaBoost.new(dataSet, 50, level)              
        print classifiers.confusion        
        classifiers.save("../../classifiers/eye-%2i" % i)

def run():    
    levels = [10, 10, 20, 20, 20, 20, 20, 30, 30, 35, 35, 35, 35, 50, 100, 100, 100, 200, 200, 200, 250, 500, 1000]    
    # levels = [10, 10, 10, 20]
    dataSet = [(x, l) for x, l in AdaBoost.loadSamples('../.tmp/true-eye', '../.tmp/false-eye')]
    # dataSet = [(x, l) for x, l in AdaBoost.loadSamples('../../.tmp/true-eye', '../../.tmp/false-eye')]    
    random.shuffle(dataSet)        
    for i, level in enumerate(levels):
        classifiers = AdaBoost.new(dataSet, 50, level)              
        print classifiers.confusion
        yield classifiers        


if __name__ == '__main__':
    main()    