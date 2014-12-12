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