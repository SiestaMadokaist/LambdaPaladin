import os
import json
import AdaBoostClassifier as Ada
import Helper as helper
from PIL import Image
import sys
import numpy as np
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
    for y in xrange(0, ylim - height, 10):
        for x in xrange(0, xlim - width, 10):
            yield [x, y, width, height]

def eyeFinder(classifiers, image):
    w, h = image.size        
    arr = [d for d in image.getdata()]
    sumTable = Ada.SumTable(w, h, arr)
    identified = []
    tot = 0
    xx = 0
    for area in ranges(image.size, (100, 48)):                
        args = sumTable.haar(*area)        
        test = helper.all(lambda c: c.apply(*args) > 0, classifiers, 3)
        if test:                    
            x, y, w, h = area            
            x1, y1 = x + w, y + h             
            yield x, y, x1, y1

def maskWrite(regions, origin, masktarget):
    def decision(x, y):
        def lambd(args):   
            x0, y0, x1, y1 = args
            return x0 <= x <= x1 and y0 <= y <= y1
        return lambd
    data = origin.load()
    w, h = origin.size
    for region in regions:
        x0, y0, x1, y1 = region
        for x in xrange(x0, x1):
            data[x, y0] = 255
            data[x, y1] = 255
        for y in xrange(y0, y1):
            data[x0, y] = 255
            data[x1, y] = 255

    # for y in xrange(h):
    #     for x in xrange(w):
    #         if not helper.any(decision(x, y), regions, 1):                
    #             data[x, y] = 255
def main():    
    classifiers = [classifier for classifier in classifierLoader()]    
    for num, image in enumerate(imagesLoader()):        
        immask = [[0 for __ in xrange(image.size[0])] for _ in xrange(image.size[1])]
        candidates = [eyeArea for eyeArea in eyeFinder(classifiers, image)]
        maskWrite(candidates, origin=image, masktarget=immask) 
        image.save('../out/%s.jpg' % num)

if __name__ == '__main__':
    main()