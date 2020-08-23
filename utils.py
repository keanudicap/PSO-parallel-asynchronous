import math
from functools import reduce
import random
import numpy as np

def euclideanDistance(point1, point2):
    return math.sqrt(
        reduce(
            (lambda memo, pair: memo + (pair[1] - pair[0]) ** 2), zip(point1, point2), 0
        )
    )


def regroup(swarm, swarm_size):
    random.shuffle(swarm)
    subswarms = [swarm[i:i+swarm_size] for i in range(0,len(swarm),swarm_size)]
    return subswarms

def sa_getnew(x, n_dims, T):
    u = np.random.uniform(-1, 1, size=n_dims)
    x_new = x + 20 * np.sign(u) * T * ((1 + 1.0 / T) ** np.abs(u) - 1.0)
    return x_new

def check_convergence(best_pos):
    for ipos in best_pos:
        if abs(ipos)>1e-3:
            return False
    return True