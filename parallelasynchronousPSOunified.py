import random
from mpi4py import MPI
from functions import *
from collections import deque
import time
import math
from scipy.spatial import cKDTree
import numpy as np
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

    def update_unified_velocity(self, best_pos_l, best_pos_g):
        w = 0.5
        c1 = 2
        c2 = 2
        u=0.5

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.best_pos_in[i] - self.position[i])
            vel_social_local = c2 * r2 * (best_pos_l[i] - self.position[i])
            vel_social_global=c2 * r2 * (best_pos_g[i] - self.position[i])
            self.velocity[i] = u*(w * self.velocity[i] + vel_social_global + vel_cognitive)+\
                            (1-u)*(w * self.velocity[i] + vel_social_local + vel_cognitive)

    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position[i] += self.velocity[i]

            if self.position[i] < bounds[0]:
                self.position[i] = bounds[0]

            if self.position[i] > bounds[1]:
                self.position[i] = bounds[1]


class ParallelasynchronousPSOunified():
    def __init__(self, num_d, bounds, num_particles, num_iter):

        global num_dimensions
        num_dimensions = num_d

        best_cost_g = float('inf')
        best_cost_l=float('inf')
        best_pos_l = []
        best_pos_g = []

        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(bounds))

        k = 10
        positions=[]
        for iswarm in swarm:
            positions.append(iswarm.position)
        positions=np.array(positions)
        tree = cKDTree(positions)

        evalQueue = deque(range(num_particles))

        for i in range(1, size):
            p = evalQueue.popleft()
            obj_comm = (p, swarm[p].position)
            comm.send(obj_comm, dest=i)

        for i in range(num_iter):
            count=0
            while(count<=num_particles):
                obj_recv = comm.recv(source=MPI.ANY_SOURCE, status=status)
                id_recv = obj_recv[0]
                f_recv = obj_recv[1]
                src_rank = status.Get_source()
                evalQueue.append(id_recv)

                id_eval=evalQueue.popleft()
                count+=1
                # print(f'id is {id_eval}')

                swarm[id_recv].cost = f_recv
                if f_recv < swarm[id_recv].best_cost_in:
                    swarm[id_recv].best_pos_in = list(swarm[id_recv].position)
                    swarm[id_recv].best_cost_in = float(f_recv)

                if f_recv < best_cost_g:
                    best_cost_g = float(f_recv)
                    best_pos_g = list(swarm[id_recv].position)

                _, ii = tree.query(np.array(swarm[id_recv].position), k=range(1, k), p=2)
                for ni in ii:
                    if swarm[ni].best_cost_in < best_cost_l:
                        best_cost_l=float(swarm[ni].best_cost_in)
                        best_pos_l = list(swarm[ni].position)

                swarm[id_eval].update_unified_velocity(best_pos_l, best_pos_g)
                swarm[id_eval].update_position(bounds)

                if len(evalQueue) != 0:
                    obj_comm = (id_eval, swarm[id_eval].position)
                    comm.send(obj_comm, dest=src_rank)
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
print(f'rank is {rank} and size is {size}')
# num_d = [10, 20, 30]
# p = [40, 120, 200]
num_d = [30]
p = [200]
costfun = sphere
bounds = (0, 10)

for id in num_d:
    for ip in p:
        if rank == 0:
            start_time = time.time()
            ParallelasynchronousPSOunified(id, bounds, num_particles=ip, num_iter=1000)
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



