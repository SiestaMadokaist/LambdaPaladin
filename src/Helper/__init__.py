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
        r = func(item)        
        if r:            
            tolerance -= 1
        if not tolerance:
            return True        
    return False

def all(func, iterable, tolerance=1):    
    for item in iterable:
        r = func(item)        
        if not r:
            tolerance -= 1            
        if not tolerance:
            return False
    return True

def all2(fiterable, tolerance=1):
    for f, item in fiterable:
        print f(item)
        if not f(item):
            return False 
    return True