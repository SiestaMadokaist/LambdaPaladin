import os
import json
import AdaBoostClassifier as Ada
import Helper as helper
from PIL import Image
import sys
import numpy as np
import CornerDetection as cd
import time
import math
@classmethod
def loadClassifiers(cls, filepath):
    def loader():
        for element in js:
            alpha, paladinjs = helper.unpackJS(element, jskeys)
            N, origin, const = helper.unpackJS(paladinjs, paladinkeys)
            paladin = Ada.Paladin(N, origin, const)                        
            yield alpha, paladin

    js = json.load(open(filepath, 'rb'))
    jskeys = ["Alpha", "Paladin"]
    paladinkeys = ["N", "origin", "const"]
    alphas, paladins = zip(*loader())
    return Ada.AdaBoost(alphas=alphas, paladins=paladins)    

Ada.AdaBoost.loadClassifiers = loadClassifiers

def classifierLoader(classifierDirectory='../classifiers'):    
    for fi in os.listdir(classifierDirectory):
        if not fi.startswith("eye"):
            continue
        path = "%s/%s" % (classifierDirectory, fi)
        classifiers = Ada.AdaBoost.loadClassifiers(path)
        yield classifiers

def imagesLoader(imagesDirectory='../dataset/jaffe'):
    for fi in os.listdir(imagesDirectory):
        path = "%s/%s" % (imagesDirectory, fi)
        image = Image.open(path)
        yield image

def ranges(size, space):
    xlim, ylim = size
    width, height = space
    for y in xrange(0, ylim - height, 6):
        for x in xrange(0, xlim - width, 6):
            yield [x, y, width, height]

def eyeFinder(classifiers, image):
    w, h = image.size        
    arr = [d for d in image.getdata()]
    sumTable = Ada.SumTable(w, h, arr)
    identified = []
    tot = 0
    xx = 0
    for area in ranges(image.size, (106, 48)):                
        args = sumTable.haar(*area)                
        test = helper.all(lambda c: c.apply(*args) > 0, classifiers, 3)
        if test:                    
            x, y, w, h = area            
            x1, y1 = x + w, y + h             
            yield x, y, x1, y1

def maskWrite(regionCandidates, origin):
    def decision(x, y):
        def lambd(args):   
            x0, y0, x1, y1 = args
            return x0 <= x <= x1 and y0 <= y <= y1
        return lambd
    data = origin.load()
    w, h = origin.size
    for region in regionCandidates:
        x0, y0, x1, y1 = region
        for x in xrange(x0, x1):
            data[x, y0] = 255
            data[x, y1] = 255
        for y in xrange(y0, y1):
            data[x0, y] = 255
            data[x1, y] = 255

def naiveBayes(avg, stddev):
    variance = math.sqrt(stddev)
    @orderOfLog10
    def aply(x):    
        divisor = math.sqrt(2 * math.pi * stddev * stddev)
        diff = x - avg
        exponetor = - (diff * diff) / (2 * stddev * stddev)
        return math.exp(exponetor)/divisor
    return aply

def orderOfLog10(func):
    return lambda *args: math.log(func(*args), 10)

def main(): 
    def ftrs(args):
        x, y, reg = args        
        return nbx(x) > -3 and nby(y) > -3        
        
    avg = [374.88853020859023, 1014.8650253216985]
    std = [56.422630355577567, 61.952098016350945]        
    nbs = map(naiveBayes, avg, std)
    nbx, nby = nbs
    classifiers = [classifier for classifier in classifierLoader()]        
    for num, image in enumerate(imagesLoader()):        
        regionCandidates = [region for region in eyeFinder(classifiers, image)]        
        crops = [image.crop(region) for region in regionCandidates]
        eigs = map(cd.detect, crops)                
        xs, ys = zip(*eigs)
        filtered = [region for x, y, region in filter(ftrs, zip(xs, ys, regionCandidates))][:4]        
        if not filtered: continue
        bounds = zip(*filtered)        
        boxLimit = [f(*args) if len(args) > 1 else args[0] for f, args in zip((min, min, max, max), bounds)]                
        hello = image.crop(boxLimit)
        hello.save('fc.jpg')

if __name__ == '__main__':
    main()     