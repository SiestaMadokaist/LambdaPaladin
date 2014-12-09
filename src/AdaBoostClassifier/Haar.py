# Implementation of SumTable to support Haar
# reference: http://en.wikipedia.org/wiki/Summed_area_table
from PIL import Image
import os
import sys
import random

class SumTable(object):    
    @classmethod
    def picks(cls, arr, x, y):
        if(x < 0 or y < 0): 
            return 0
        else:
            return arr[y][x]

    def __init__(self, w, h, arr):
        self.width = w;
        self.height = h;
        self.arr = arr;
        self.arrsum = self.generateSum()    

    def __getitem__(self, args):        
        x, y, falseW, falseH = args
        w = min(self.width - x, falseW)
        h = min(self.height - y, falseH)
        # handle for out of bound exception.
        a = SumTable.picks(self.arrsum, x + w - 1, y - 1)
        b = SumTable.picks(self.arrsum, x - 1, y + h - 1)
        c = SumTable.picks(self.arrsum, x - 1, y - 1)
        d = SumTable.picks(self.arrsum, x + w - 1, y + h -1)                        
        out = d - a - b + c        
        return out    

    def get(self, x, y):
        return self.arr[y * self.width + x]

    def generateSum(self):
        def getNewVal(x, y, arr):
            if(x == 0 and y == 0): return self.get(x, y)
            elif(y == 0): return arr[y][x-1] + self.get(x, y)
            elif(x == 0): return arr[y-1][x] + self.get(x, y)
            else: return arr[y-1][x] + arr[y][x-1] - arr[y-1][x-1] + self.get(x, y)
        myvar = [[] for _ in xrange(self.height)]
        for y in xrange(self.height):
            for x in xrange(self.width):                
                myvar[y].append(getNewVal(x, y, myvar))            
        return myvar    

    def haar(self, *boundaries):                
        x, y, w0, h0 = boundaries        
        def getA():
            w = w0 / 2
            h = h0
            part1 = self[x, y, w, h]
            part2 = self[x + w, y, w, h]
            return part1 - part2

        def getB():
            w = w0 / 4
            h = h0
            part1 = self[x, y , w, h]
            part2 = self[x + w, y, w * 2, h]
            part3 = self[x + 3 * w, y, w, h]
            return part1 - part2 + part3

        def getC():
            w = w0
            h = h0 / 2
            part1 = self[x, y, w, h]
            part2 = self[x, y + h, w, h]
            return part1 - part2

        def getD():
            w = w0 / 2
            h = h0 / 2
            part1 = self[x, y, w, h]
            part2 = self[x + w, y, w, h]
            part3 = self[x, y + h, w, h]
            part4 = self[x + w, y + h, w, h]
            return part2 - part3

        return [getA(), getB(), getC(), getD()]

def main():    
    
    mySumTable = SumTable(10, 10, range(100))    
    print mySumTable.haar(0, 0, 3, 3)

if __name__ == '__main__':
    main()