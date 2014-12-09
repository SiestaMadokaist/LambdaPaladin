# Adaboost implementation taken from https://gist.github.com/tristanwietsma/5486024 with a few change.

from __future__ import division
from Paladin import Paladin
import numpy as np
import sys
import random
import json

def getSeeds(n):
    return [random.randint(-10, 10) for _ in xrange(n)]

class AdaBoost(object):
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
    
    def __init__(self, trainSet, validationSet):
        self.TrainingSet = trainSet
        self.ValidationSet = validationSet
        self.N = len(self.TrainingSet)
        self.Weights = np.ones(self.N)/self.N
        self.Alpha = []
        self.Paladins = []

    @property
    def accuracy(self):
        targetSet, expectationSet = zip(*self.validationSet)
        return sum(e for e in self.evaluate(targetSet, expectationSet)) / len(targetSet)

    def updated_weights(self, errors, weights, alpha):
        for error, weight in zip(errors, weights):
            exponetor = alpha if error else -alpha
            yield weight * np.exp(exponetor)

    def add_paladin(self, paladin):
        errors = np.array([expectation != paladin.rule(*args) for args, expectation in self.TrainingSet])
        e = (self.Weights * errors).sum()
        alpha = 0.5 * np.log((1 - e) / e)
        w = np.array([weight for weight in self.updated_weights(errors, self.Weights, alpha)])
        self.Weights = w / w.sum()        
        self.Alpha.append(alpha)
        self.Paladins.append(paladin)

    def apply(self, *args):        
        hx = [alpha * paladin.rule(*args) for alpha, paladin in zip(self.Alpha, self.Paladins)]
        r = np.sign(sum(hx))
        return r

    def evaluate(self, targetSet, expectationSet):
        for args, expect in zip(targetSet, expectationSet):
            yield self.apply(*args) == expect

    @property    
    def confusion(self):
        targetSet, expectationSet = zip(*self.ValidationSet)
        memo = [[0, 0], [0, 0]]
        for args, e in zip(targetSet, expectationSet):
            r = self.apply(*args)            
            memo[e == -1][r == -1] += 1
        return memo


    def __str__(self):
        return json.dumps([{
            "Paladin": {
                "origin" : paladin.origin,
                "const": paladin.const
                },
            "Alpha": alpha
        } for paladin, alpha in zip(self.Paladins, self.Alpha)])

def main():
    levels = [10, 10, 20, 20, 20, 20, 20, 30, 30, 35, 35, 50, 100, 250, 500, 1000]
    dataSet = [(x, l) for x, l in AdaBoost.loadSamples('../.tmp/false-eye', '../.tmp/true-eye')]
    random.shuffle(dataSet)
    for i, level in enumerate(levels):
        classifiers = AdaBoost.new(dataSet, 50, level)      
        print classifiers.confusion        

if __name__ == '__main__':
    main()