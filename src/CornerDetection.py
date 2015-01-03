from PIL import Image
import Convolution
import Helper as helper
import operator as op
import numpy as np
def eigen(image, debug=False):
    return detect(image, debug=debug)

def detect(image, threshold=100, debug=False):    
    def applyMask(kernel):
        mask = Convolution.maskKernel(kernel)        
        return Convolution.convolute(image, mask, (3, 3))

    def norm(v):
        return v / 255.0

    def view(f):
        return int(f * 100) + 100

    def showDerivationCovariant(Ix, Iy, size):    
        fields = [[0 for i in xrange(size)] for j in xrange(size)]        
        for x, y in zip(Ix, Iy):        
            fields[view(y)][view(x)] = 255        
        helper.arr2dimage(fields, "img/hello.jpg")

    
    kernelX = [(i % 3) - 1 for i in xrange(9)]    
    kernelY = [(i / 3) - 1 for i in xrange(9)]

    Ix = map(norm, applyMask(kernelX))
    Iy = map(norm, applyMask(kernelY))

    if debug: showDerivationCovariant(Ix, Iy, 201)

    Eigen = map(sum, ([
            map(op.mul, Ix, Ix),
            map(op.mul, Ix, Iy),
            map(op.mul, Iy, Ix),
            map(op.mul, Iy, Iy)        
        ])  
    )
    eigen =  np.linalg.eig([_ for _ in helper.group(Eigen, 2)])
    return eigen[0]

if __name__ == '__main__':
    path = '../dataset/jaffe/KA.AN1.39.jpg'    
    image = Image.open(path).crop((74, 103, 123, 145))
    print detect(image, threshold=100, debug=True)