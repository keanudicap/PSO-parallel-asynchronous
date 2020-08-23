import random
from mpi4py import MPI
from collections import deque
import time
import math
from functions import *
from utils import *

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

    def update_local_velocity(self, best_pos_l):
        w = 0.5
        c1 = 2
        c2 = 2

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.best_pos_in[i] - self.position[i])
            vel_social = c2 * r2 * (best_pos_l[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + vel_social + vel_cognitive

    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position[i] += self.velocity[i]

            if self.position[i] < bounds[0]:
                self.position[i] = bounds[0]

            if self.position[i] > bounds[1]:
                self.position[i] = bounds[1]

class SequentialPSODMS():
    def __init__(self, num_d, bounds, num_particles, num_iter, costFunction):
        global num_dimensions
        num_dimensions=num_d

        sub_swarm_num = 5
        R=10
        swarm_size=int(num_particles/sub_swarm_num)
        best_cost_l = [float('inf')] * sub_swarm_num
        best_pos_l = [[float('inf') for i in range(num_dimensions)] for j in range(sub_swarm_num)]
        best_pos_g = []
        best_cost_g = float('inf')
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(bounds))

        subswarms = regroup(swarm, swarm_size)

        for i in range(0, num_iter):
            if i%R==0:
                best_cost_l = [float('inf')] * sub_swarm_num
                best_pos_l = [[float('inf') for i in range(num_dimensions)] for j in range(sub_swarm_num)]
                subswarms=regroup(swarm,swarm_size)
            for ind, subswarm in enumerate(subswarms):
                for j in range(len(subswarm)):
                    f = costFunction(subswarm[j].position)

                    if f < subswarm[j].best_cost_in:
                        subswarm[j].best_cost_in = f
                        subswarm[j].best_pos_in = subswarm[j].position

                    if f < best_cost_g:
                        best_cost_g = float(f)
                        best_pos_g = list(subswarm[j].position)

                    if f < best_cost_l[ind]:
                        best_cost_l[ind] = float(f)
                        best_pos_l[ind] = list(subswarm[j].position)

                for j in range(len(subswarm)):
                    subswarm[j].update_local_velocity(best_pos_l[ind])
                    subswarm[j].update_position(bounds)

                if check_convergence(best_pos_g):
                    break
        print(f'Best position : {best_pos_g}')
        print(f'Best cost : {best_cost_g}')


def main():
    start_time = time.time()

    num_d = [10,20,30]
    p=[40,120,200]
    costfun = sphere
    bounds = (0, 10)

    for id in num_d:
        for ip in p:
            SequentialPSODMS(id, bounds, num_particles=ip, num_iter=400, costFunction=costfun)
            print(f"time taken: d is {id} and p is {ip}")
            print(f'{time.time() - start_time:.2e}')

if __name__ == "__main__":
    main()

