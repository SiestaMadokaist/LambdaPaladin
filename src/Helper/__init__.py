from PIL import Image
def deprecated(message):
    def decorator(func):        
        def aply(*args):
            print message            
            return func(*args)
        return aply
    return decorator

def unpackJS(j, keys):
    return [j[key] for key in keys]

def unpack2(arg, *rest):
    return arg, rest

def unpack3(arg1, arg2, *rest):
    return arg1, arg2, rest

def map(func, iterable):
    for i in iterable:
        yield func(i)

def any(func, iterable, tolerance=1):
    for item in iterable:
        r = func(*item)        
        if r:            
            tolerance -= 1
        if not tolerance:
            return True        
    return False

def all(func, iterable, tolerance=1):        
    for item in iterable:
        if isinstance(item, (tuple, list)):
            r = func(*item)
        else:
            r = func(item)        
        if not r:
            tolerance -= 1            
        if not tolerance:
            return False
    return True

def all2(fiterable, tolerance=1):
    for f, item in fiterable:        
        if not f(*item):
            return False 
    return True

def group(items, n):
    items = [_ for _ in items]    
    l = (len(items) / n)  * n
    for i in xrange(0, l, n):
        yield items[i:i+n]

@deprecated("arr2image is deprecated, please use arr2dimage")
def arr2image(arr, size, fout):
    im = Image.new("L", size)
    iml = im.load()
    for y, line in enumerate(arr):
        for x, val in enumerate(line):            
            iml[x, y] = val
    im.save(fout)

def arr2dimage(arr, fout):
    size = (len(arr[0]), len(arr))
    im = Image.new("L", size)
    iml = im.load()
    for y, line in enumerate(arr):
        for x, val in enumerate(line):            
            iml[x, y] = val
    im.save(fout)

def generator1dimage(gen, size, fout):    
    w, h = size        
    image = Image.new("L", size)
    im = image.load()
    for y in xrange(h):
        for x in xrange(w):
            try:
                im[x, y] = gen.next()
            except:
                break
    image.save(fout)