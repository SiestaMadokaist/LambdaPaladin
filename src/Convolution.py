import math
import numpy as np
from PIL import Image
import sys
import code
import operator as op
import Helper as helper
# generator yang ngambil area 3x3 disekitar p, q
def pick(im, xx, yy, size=(3, 3)):
    w, h = size
    bound = lambda b: lambda x: (- (b / 2) + x, (b + 1) / 2 + x)
    wBound = bound(w)    
    hBound = bound(h)    
    for y in xrange(*hBound(yy)):
        for x in xrange(*wBound(xx)):
            yield im[x, y]

# generator buat ngetransform gambar, jadi list of area3x3 buat dikonvolusi
def areaGenerator(image, size=(3, 3)):
    im = image.load()
    w, h = image.size    
    for y in xrange(0, h):
        for x in xrange(0, w):
            yield pick(im, x, y, size)

def monadError(func, default):
    def monadic(*args):
        try:
            r = func(*args)                        
        except:
            return default        
        else:         
            return r
    return monadic

def maskKernel(kernel):    
    def aply(values):
        return sum(m * v for m, v in zip(kernel, values))        
    return aply

def convolute(image, mask, size):        
    monadMask = monadError(mask, 255)
    for area in areaGenerator(image, size):        
        r = monadMask(area)
        if r < -255: yield -255
        elif r > 255: yield 255
        else: yield r

class LDP(object):
    @classmethod
    def Directions(cls, i):
        idx = i % 8
        return [5, 8, 7, 6, 3, 0, 1, 2][idx]

    @classmethod
    def KirschGenerator(cls, direction):
        fives = map(LDP.Directions, xrange(direction-1, direction+2))        
        for y in xrange(3):
            for x in xrange(3):
                if x == y == 1:
                    yield 0
                else:
                    yield 5 if y * 3 + x in fives else -3

    @classmethod
    def Kirschedge(cls, direction):        
        if direction != all:
            return LDP.KirschGenerator(direction)
        else:
            return map(LDP.KirschGenerator, xrange(8))

    @classmethod    
    def Mask(cls, direction=all):
        if direction != all:
            kernel = [_ for _ in LDP.Kirschedge(direction)]
            return maskKernel(kernel)
        else:                        
            kernels = [maskKernel([e for e in m]) for m in LDP.Kirschedge(all)]
            def aply(area):
                memo = [e for e in area]
                result = [kernel(memo) for kernel in kernels]
                return result
                sortedOut = sorted(enumerate(result), key=lambda x: x[1], reverse=True)                
                xs = (k for k, v in sortedOut[:3])
                return sum(2 ** x for x in xs) / 2
            return aply

    def __init__(self, size):
        w, h = size
        self.w = w - 2
        self.h = h - 2
        self.pointer = 0
        self.items = [[] for _ in xrange(8)]                

    def apply8(self, area):        
        kernel0 = LDP.Kirschedge(0)
        masker = maskKernel(kernel0)
        memo = list(area)
        output0 = sum(a * m for a, m in zip(kernel0, memo))
        currentResult = output0        
        for i in xrange(0, 8):
            self.items[i].append(currentResult)
            toNeg = [5, 8, 7, 6, 3, 0, 1, 2][i - 1]
            fromNeg = [5, 8, 7, 6, 3, 0, 1, 2][i + 2]
            prevResult = currentResult
            currentResult = prevResult - (8 * memo[toNeg]) + (8 * memo[fromNeg])                        
        self.pointer += 1
        return 0

    def save(self, basename):
        assert len(self.items[0]) == self.w * self.h
        for i, data in enumerate(self.items):
            fout = "%s-%s.jpg" % (basename, i)
            image = Image.new("L", (self.w, self.h))            
            image.putdata(data)            
            image.save(fout)

    
if __name__ == '__main__':    
    image = Image.open('img/paav.jpg')
    ldpMonad = LDP(image.size)    
    list(convolute(image, ldpMonad.apply8, (3,3)))
    ldpMonad.save('img/test')
    