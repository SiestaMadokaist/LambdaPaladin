import math
from PIL import Image
import sys
import code
import operator as op
import profile
import random
# import Helper as helper
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
        return [5, 2, 1, 0, 3, 6, 7, 8][idx]

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
                sortedOut = sorted(enumerate(result), key=lambda x: x[1], reverse=True)                
                xs = (k for k, v in sortedOut[:3])
                return sum(2 ** x for x in xs) / 2
            return aply

def FastLDPTransform(fn, size):
    w, h = size
    kernel0 = list(LDP.Kirschedge(0))
    rotation = [5, 2, 1, 0, 3, 6, 7, 8, 5, 2, 1, 0, 3, 6, 7, 8]
    eights = range(8)
    def calculateAt(x, y):        
        assert 1 <= x < w-1, 'out of bound at x'
        assert 1 <= y < h-1, 'out of bound at y'
        values = [fn[a * w + b] for a in xrange(y - 1, y + 2) for b in xrange(x - 1, x + 2)]        
        streams = [sum(m * v for m, v in zip(kernel0, values))]        
        for i in xrange(7):
            toNeg = rotation[i - 1]
            fromNeg = rotation[i + 2]
            prev = streams[-1]
            streams.append(prev - (8 * values[toNeg]) + (8 * values[fromNeg]))
        tmp = sorted(eights, key = streams.__getitem__, reverse=True)        
        def atLevel(k):                                    
            return sum(2 ** i for i in tmp[:k])
        return atLevel
    return calculateAt

def LDPTransform(fn, size, x, y, k):
    w, h = size    
    values = [fn[a * w + b] for a in xrange(y - 1, y + 2) for b in xrange(x - 1, x + 2)]        
    streams = [sum(m * v for m, v in zip(kernel, values)) for kernel in kernels]        
    tmp = sorted(eights, key = streams.__getitem__, reverse=True)
    return sum(2 ** i for i in tmp[:k])

def testLDP():
    fn = range(65536)
    return [LDPTransform(fn, (256, 256), x, y, 3) for x in xrange(1, 255) for y in xrange(1, 255)]

def testFLDP():
    fn = range(65536)
    FLDP = FastLDPTransform(fn, (256, 256))    
    return [FLDP(x, y)(3) for x in xrange(1, 255) for y in xrange(1, 255)]

if __name__ == '__main__':    
    a = testFLDP()
    # b = testLDP()    