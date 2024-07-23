import math
import time
import numpy as np
from simulator import Simulator
from infrastructure import Infrastructure
from knapsack import Knapsack
import random
from IDM import IDM
from scipy.interpolate import CubicSpline
import heuristic_model

def make_path(start, goal):
    m = (goal[1] - start[1]) / (goal[0] - start[0])
    straight = np.array([1, m]) / np.linalg.norm(np.array([1, m])) * np.sign(goal - start)
    perp = np.array([1, -1/m]) / np.linalg.norm(np.array([1, -1/m]))
    points = []
    points.append(start)
    p = start.copy()
    n = random.randint(3, 7) # TODO: tune
    delta = np.linalg.norm(goal - start) / n
    for i in range(n - 1): # TODO: tune
        p += straight * delta
        eps = random.uniform(-3, 3) # TODO: tune
        pt = p + perp * eps
        points.append(pt)
    points.append(goal)
    points = np.array(sorted(sorted(points,key=lambda e:e[1]),key=lambda e:e[0]))
    if points[0][0] > points[1][0]:
        points.reverse()
    spline = CubicSpline([x[0] for x in points], [x[1] for x in points])

    # parameterize spline
    path = {0: start}
    dx = 0.1
    t = 0
    dt = 0.1
    x = min(start[0], goal[0])
    arc_len = 0
    len_max = 1 # TODO: tune; length traveled in dt seconds
    while x < max(goal[0], start[0]):
        x += dx
        arc_len += math.sqrt(1 + spline.derivative()(x)**2) * dx
        if arc_len >= len_max:
            arc_len = 0
            t += dt
            path[round(t, 1)] = np.array([x, spline(x)])
    path[round(t + dt, 1)] = goal

    return path

def run_sim(model):

    m = 1
    n = 1

    # C = [[random.randint(1, 10) for j in range(n)] 
    #         for i in range(n)]

    C = [[1 for j in range(n)] for i in range(n)]
    beta = random.randint(1, n)

    x = {'pH': [], 'vH': [], 'aH': [], 'gH': [], 'path': [],
        'pR': [], 'vR': [], 'aR': [], 'gR': [],
        'walls': []}

    start = time.time()

    for i in range(m):
        x['pH'].append(np.array([random.uniform(-100, 100), random.uniform(-100, 100)]))
        x['gH'].append(np.array([random.uniform(-100, 100), random.uniform(-100, 100)]))
        x['path'].append(make_path(x['pH'][i], x['gH'][i]))
        x['vH'].append(np.array([0., 0.]))
        x['aH'].append(np.array([0., 0.]))

    for i in range(n):
        x['pR'].append(np.array([random.uniform(-100, 100), random.uniform(-100, 100)]))
        x['gR'].append(np.array([random.uniform(-100, 100), random.uniform(-100, 100)]))
        x['vR'].append(np.array([0., 0.]))
        x['aR'].append(np.array([0., 0.]))

    num_walls = 10
    for i in range(num_walls):
        p1 = np.array([random.uniform(-100, 100), random.uniform(-100, 100)])
        eps = 50
        p2 = np.array([p1[0] + random.uniform(-eps, eps), p1[1] + random.uniform(-eps, eps)])
        x['walls'].append((p1, p2))

    t_max = 100
    dt = 0.1

    sim = Simulator(x, m, n, dt)
    infra = Infrastructure(sim)
    sim.add_infra(infra)
    knap = Knapsack(model, sim, infra, C, beta)

    while sim.t <= t_max:

        # iterate simulation
        x = sim.move()
        sim.vis()

        # run individual sensing
        for r in range(n):
            for h in range(m):
                infra.sense(r, h)

        # if sim.t % 5 == 0:
        #     # run knapsack to decide communications
        #     comms = [(a, b, h) for a in range(n) for b in range(n) if a != b for h in range(m)] # consider all potential comms
        #     chosen = knap.sack(comms)[1] # indices of chosen comms

        #     # execute communications
        #     for i in chosen:
        #         (a, b, h) = comms[i]
        #         infra.share(a, b, h)

        # input()

    end = time.time()
    print("RUNTIME", end - start)

    # return loss # TODO

if __name__ == '__main__':
    run_sim(heuristic_model.heuristic)