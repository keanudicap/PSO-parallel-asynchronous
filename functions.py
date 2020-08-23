import numpy as np
import math

def sphere(x):
    return sum([y**2 for y in x])

def rosenbrock(x):
    res=0
    for id in range(1,len(x)):
        res+=100*((x[id]**2 - x[id-1])**2) + (x[id-1] - 1)**2
    return res

def ackley(x):
    return -20 * np.exp(-0.2 * (sum([y**2 for y in x])/len(x) ** 0.5)) - \
          np.exp(sum([np.cos(2 * math.pi * y)/len(x) for y in x])) + 20 + math.e

def griewank(x):
    return sum([y**2 / 4000 for y in x]) - np.prod([np.cos(y/((i+1)**0.5)) for i, y in enumerate(x)]) + 1

def rastrigin(x):
    return sum([y**2 - 10*np.cos(2*math.pi*y) + 10 for y in x])

def schwefel(x):
    a = 418.9829
    return a*len(x)-sum([y*math.sin(math.sqrt(abs(y))) for y in x])

