import os
import json
import AdaBoostClassifier as Ada
import Helper as helper
from PIL import Image
import sys
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
            yield area            

def main():    
    classifiers = [classifier for classifier in classifierLoader()]    
    for num, image in enumerate(imagesLoader()):        
        for eyeArea in eyeFinder(classifiers, image):
            print num, eyeArea
            
if __name__ == '__main__':
    main()