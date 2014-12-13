import profile as p 
import operator as op

def main():
    inf = 10 ** 6
    # add = lambda x, y:  x + y
    # map(op.add, xrange(inf), xrange(inf))    
    # map(lambda x, y:  x + y, xrange(inf), xrange(inf))    
    [x + y for x, y in zip(xrange(inf), xrange(inf))],

p.run("main()")