class Paladin(object):    
    def __init__(self, N, origin, const):        
        self.origin = origin        
        self.seeds = [(m, i % N) for m, i in Paladin.group(self.origin, 2)]
        self.N = N
        self.const = const

    @classmethod
    def group(self, items, n):
        l = (len(items) / n)  * n
        for i in xrange(0, l, n):
            yield items[i:i+n]        

    def rule(self, *args):
        return (sum(m * args[i] for m, i in self.seeds) > self.const) * 2 - 1

    def __str__(self):
        return json.dumps(self.seeds)
