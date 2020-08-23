import random
from mpi4py import MPI
from functions import *
from collections import deque
import time
import math
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


class ParallelPSO():
    def __init__(self, num_d, bounds, num_particles, num_iter):

        global num_dimensions
        num_dimensions = num_d

        best_cost_g = float('inf')
        best_pos_g = []

        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(bounds))

        for i in range(num_iter):

            evalQueue = deque(range(num_particles))

            # POP AND SEND PARTICLES TO EACH SLAVE NODE
            for i in range(1, size):
                p = evalQueue.popleft()
                obj_comm = (p, swarm[p].position)
                comm.send(obj_comm, dest=i)

            idle = 0
            # FURTHER LOOPING
            while (1):
                obj_recv = comm.recv(source=MPI.ANY_SOURCE, status=status)
                id_recv = obj_recv[0]
                f_recv = obj_recv[1]
                src_rank = status.Get_source()

                swarm[id_recv].cost = f_recv
                if f_recv < swarm[id_recv].best_cost_in:
                    swarm[id_recv].best_pos_in = list(swarm[id_recv].position)
                    swarm[id_recv].best_cost_in = float(f_recv)

                if f_recv < best_cost_g:
                    best_cost_g = float(f_recv)
                    best_pos_g = list(swarm[id_recv].position)

                if len(evalQueue) != 0:
                    j = evalQueue.popleft()
                    obj_comm = (j, swarm[j].position)
                    comm.send(obj_comm, dest=src_rank)
                else:
                    idle += 1

                if idle == size - 1:
                    break

            for j in range(0, num_particles):
                swarm[j].update_velocity(best_pos_g)
                swarm[j].update_position(bounds)

            if check_convergence(best_pos_g):
                break
        for k in range(1, size):
            comm.send(0, dest=k, tag=200)
        print(f'Best position : {best_pos_g}')
        print(f'Best cost : {best_cost_g}')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()

num_d = [10, 20, 30]
p = [40, 120, 200]
costfun = sphere
bounds = (0, 10)

print(f'rank is {rank} and size is {size}')

for id in num_d:
    for ip in p:
        if rank == 0:
            start_time = time.time()
            ParallelPSO(id, bounds, num_particles=ip, num_iter=400)
            print(f"time taken: d is {id} and p is {ip}")
            print(f'{time.time() - start_time:.2e}')

        else:
            while (1):
                obj_recv = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                if tag == 200:
                    break

                f = costfun(obj_recv[1])
                obj_sent = (obj_recv[0], f)
                comm.send(obj_sent, dest=0)


