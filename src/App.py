
import os
import ujson as json
import AdaBoostClassifier as Ada
import Helper as helper
import sys
import numpy as np
import CornerDetection as cd
import time
import math
import Convolution
from PIL import Image
from datetime import datetime
import itertools
import code
flatten = lambda args: list(itertools.chain(*args))
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

def ImageOpenDecorator(func):
    def aply(fname, mode='r'):        
        output = func(fname)        
        output.name = fname.split("/")[-1]
        output.path = fname
        return output
    return aply

Image.open = ImageOpenDecorator(Image.open)

def classifierLoader(classifierDirectory='../classifiers'):    
    for fi in os.listdir(classifierDirectory):
        if not fi.startswith("eye"):
            continue
        path = "%s/%s" % (classifierDirectory, fi)
        classifiers = Ada.AdaBoost.loadClassifiers(path)
        yield classifiers

def imagesLoader(imagesDirectory='../dataset/jaffe', filters=lambda fpath: True):
    for fi in filter(filters, os.listdir(imagesDirectory)):
        path = "%s/%s" % (imagesDirectory, fi)
        image = Image.open(path)        
        yield image

def ranges(size, space):
    xlim, ylim = size
    width, height = space
    for y in xrange(0, ylim - height, 10):
        for x in xrange(0, xlim - width, 10):
            yield [x, y, width, height]

def eyeFinder(classifiers, image):
    w, h = image.size        
    arr = [d for d in image.getdata()]
    sumTable = Ada.SumTable(w, h, arr)    
    for area in ranges(image.size, (100, 48)):                
        args = sumTable.haar(*area)                
        test = helper.all(lambda c: c.apply(*args) > 0, classifiers, 2)
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

def process(image, classifiers, nbx, nby):
    def ftrs(args):
        x, y, reg = args        
        nx = nbx(x) 
        ny = nby(y) 
        return (nx > -3) and (ny > -3) and (nx + ny > -5)    
    regionCandidates = [region for region in eyeFinder(classifiers, image)]            
    crops = [image.crop(region) for region in regionCandidates]        
    eigs = map(cd.detect, crops)                
    xs, ys = zip(*eigs)
    filtered = [region for x, y, region in filter(ftrs, zip(xs, ys, regionCandidates))][:4]
    if not filtered: return None
    bounds = zip(*filtered)        
    boxLimit = [f(*args) if len(args) > 1 else args[0] for f, args in zip((min, min, max, max), bounds)]                
    eyeRegion = image.crop(boxLimit)
    eyeRegion.name = image.name
    return eyeRegion    
    
def eyeRegionFinder(image, classifiers, nbx, nby):
    eyeRegion = process(image, classifiers, nbx, nby)    
    if not eyeRegion: return None    
    if len(list(eyeRegion.getdata())) > 8000: 
        eyeRegion = process(eyeRegion, classifiers, nbx, nby)
        if not eyeRegion: return None
    return eyeRegion

class NNInput:                        
    def __init__(self, level):
        self.level = level    
        self.keys = [x for x in xrange(256) if self.filtr(x)]

    def filtr(self, x):
        return sum(map(int, bin(x)[2:])) == self.level

    def histogram(self, xs):
        tmpResult = [sum(1 for x in xs if x == k) for k in self.keys]
        return map(lambda x: x / float(len(xs)), tmpResult)

def cropToPQ(image, p=2, q=2):
    ew, eh = image.size
    w, h = ew / p, eh / q
    ew, eh = w * p, h * q
    x0 = range(0, ew, w)
    x1 = map(lambda x: min(x + w, ew), x0)
    y0 = range(0, eh, h)
    y1 = map(lambda y: min(y + h, eh), y0)
    return [image.crop((p, q, r, s)) for q, s in zip(y0, y1) for p, r in zip(x0, x1)]

def ImageFactory(w, h):
    def create(xs):
        im = Image.new("L", (w, h))
        im.putdata(xs)
        return im 
    return create

def LDPDataGenerator(): 
    avg = [374.88853020859023, 1014.8650253216985]
    std = [56.422630355577567, 61.952098016350945]
    nbs = map(naiveBayes, avg, std)
    nbx, nby = nbs      
    classifiers = [classifier for classifier in classifierLoader()]    
    ldpMask = Convolution.LDP.Mask(all)
    valueSets = []
    for image in imagesLoader(): 
        foutpath = "out/%s" % image.name
        cls = image.name.split(".")[1][:2]        
        eyeRegion = eyeRegionFinder(image, classifiers, nbx, nby)
        if not eyeRegion: continue
        eyeRegion.save(foutpath)
        fldp = Convolution.FastLDPTransform(list(eyeRegion.getdata()), eyeRegion.size)
        w, h = eyeRegion.size
        factory = ImageFactory(w-2, h-2)
        almostImage = [fldp(x, y) for y in xrange(1, h-1) for x in xrange(1, w - 1)]
        LDPResults = [map(lambda f: f(k), almostImage) for k in xrange(8)]
        images = map(factory, LDPResults)
        cropedImages = map(lambda i: cropToPQ(i, 6, 2), images)
        data = [map(lambda im: list(im.getdata()), imageSet) for imageSet in cropedImages]
        histogramsUnflatten = [map(NNInput(k).histogram, datum) for k, datum in enumerate(data)]
        histograms = map(flatten, histogramsUnflatten)        
        yield histograms, cls
        
def main():        
    for i in xrange(8):
        try:
            os.remove('../LDPResult/data%s.csv' % i)
        except:
            pass

    for i, (histograms, cls) in enumerate(LDPDataGenerator()):
        for k, histogram in enumerate(histograms):                        
            fout = open('../LDPResult/data%s.csv' % k, 'ab')
            printed = "%s\n%s\n" % (cls, ', '.join(map(lambda fl: "%.3f" % fl, histogram)))
            fout.write(printed)
        print i        
    
if __name__ == '__main__':    
    main()    
from math import sqrt
def eig2(arr):
    ix2, ixiy1, ixiy2, iy2 = arr
    c = ix2 * iy2 - (ixiy1 * ixiy2)
    b = -(ix2 + iy2)    
    a = 1
    d = sqrt(b ** 2 - 4 * a * c)
    v = map(lambda x: x/(2.0 * a), (-b + d, -b - d))
    return v