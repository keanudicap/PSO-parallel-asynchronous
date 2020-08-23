import random
from mpi4py import MPI
from collections import deque
import time
import math
from functions import *
from utils import *

# # ARTIFICIALLY EXPENSIVE ALPINE OBJECTIVE FUNCTION
# def costFunction(x):
#     a = 0
#     time.sleep(0.01)  # to make this artificially expensive
#     for i in x:
#         a += abs((math.sin(i) * i) + (0.1 * i))
#     return a


class Particle:
    def __init__(self, bounds):
        self.position = []
        self.velocity = []
        self.best_pos_in = []
        self.best_cost_in = float('inf')
        self.cost = float('inf')

        for i in range(0, num_dimensions):
            self.velocity.append(random.uniform(-1, 1))
            self.position.append(random.uniform(bounds[0], bounds[1]))

        self.best_pos_in = list(self.position)

    def update_velocity(self, best_pos_g):
        w = 0.5
        c1 = 2
        c2 = 2

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.best_pos_in[i] - self.position[i])
            vel_social = c2 * r2 * (best_pos_g[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + vel_social + vel_cognitive

    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position[i] += self.velocity[i]

            if self.position[i] < bounds[0]:
                self.position[i] = bounds[0]

            if self.position[i] > bounds[1]:
                self.position[i] = bounds[1]


class SequentialasynchronousPSO():
    def __init__(self, num_d, bounds, num_particles, num_iter, costFunction):
        global num_dimensions
        num_dimensions=num_d
        best_cost_g = float('inf')
        best_pos_g = []
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(bounds))

        for i in range(num_iter):
            for j in range(num_particles):
                f = costFunction(swarm[j].position)

                if f<swarm[j].best_cost_in:
                    swarm[j].best_cost_in=f
                    swarm[j].best_pos_in = swarm[j].position

                if f < best_cost_g:
                    best_cost_g = float(f)
                    best_pos_g = list(swarm[j].position)

                swarm[j].update_velocity(best_pos_g)
                swarm[j].update_position(bounds)

            # for j in range(num_particles):
            #     swarm[j].update_velocity(best_pos_g)
            #     swarm[j].update_position(bounds)
            if check_convergence(best_pos_g):
                break
        print(f'Best position : {best_pos_g}')
        print(f'Best cost : {best_cost_g}')


def main():
    start_time = time.time()

    num_d = [10,20,30]
    p=[40,120,200]
    costfun = sphere

    for id in num_d:
        for ip in p:
            bounds = (0, 10)
            SequentialasynchronousPSO(id, bounds, num_particles=ip, num_iter=400, costFunction=costfun)
            print(f"time taken: d is {id} and p is {ip}")
            print(f'{time.time() - start_time:.2e}')

if __name__ == "__main__":
    main()

