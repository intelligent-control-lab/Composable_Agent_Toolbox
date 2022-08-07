import math
from queue import Queue

import numpy as np
from scipy.spatial import KDTree

# SP = [-3.5, -3, -0.5, 0.5, 3, 3.4] # must be sorted and unique, also need to figure out how to deal with overlaps
# r = 1

# def compute_dv(s1, s2, overlap=False):
#     mag = r + r
#     if overlap:
#         mag += abs(s1 - s2)
#         print(mag)
#     else:
#         mag -= abs(s1 - s2)
#     if s1 < s2:
#         return -mag
#     return mag

# def intersects(s1, s2):
#     return abs(s1 - s2) < r + r

# def sweep(s_cur, v_cur):
#     # return the first sphere s_cur intersects when translating along v_cur, 
#     # or None if no intersection
#     # s_cur + v_cur is s_cur translated across v_cur
#     i = SP.index(s_cur)
#     if i == 0 or i == len(SP) - 1:
#         return None
#     if v_cur < 0 and intersects(s_cur + v_cur, SP[i - 1]):
#         return SP[i - 1]
#     if v_cur > 0 and intersects(s_cur + v_cur, SP[i + 1]):
#         return SP[i + 1]
#     return None

# def simulate(s1, s2):
#     outstanding = []
#     q = Queue()
#     v1 = compute_dv(s1, s2)
#     v2 = compute_dv(s2, s1)
#     q.put((s1, v1))
#     q.put((s2, v2))
#     while not q.empty():
#         s_cur, v_cur = q.get()
#         outstanding.append((s_cur, v_cur))
#         s_new = sweep(s_cur, v_cur)
#         if s_new is not None:
#             ol = abs(s_new) < abs(s_cur) + abs(v_cur)
#             v_new = compute_dv(s_new, s_cur + v_cur, overlap=ol)
#             q.put((s_new, v_new))
#     return outstanding

# if __name__ == '__main__':
#     outstanding = simulate(SP[2], SP[3]) # begin with two intersecting spheres
#     print(outstanding)






spheres = []
tree = None # will need dynamic structure (insert/delete), perhaps ikd-tree (https://github.com/hku-mars/ikd-Tree)
r = 1 # try making heterogeneous in future

class Sphere:
    def __init__(self, pos, rad):
        self.pos = pos
        self.rad = rad

def compute_dv(s1, s2):
    mag = s1.rad + s2.rad - math.sqrt(
        (s1.pos[0] - s2.pos[0])**2 + (s1.pos[1] - s2.pos[1])**2 + (s1.pos[2] - s2.pos[2])**2)
    uv = (s1.pos - s2.pos) / np.linalg.norm(s1.pos - s2.pos)
    return mag * uv

def intersects(s1, s2):
    return math.sqrt(
        (s1.pos[0] - s2.pos[0])**2 + (s1.pos[1] - s2.pos[1])**2 + (s1.pos[2] - s2.pos[2])**2
    ) < s1.rad + s2.rad

# TODO: figure out: should sweep stop at first intersection or continue thru?
# is it possible to iterate by distance 2r instead of r? (prove all intersections are detected)
def sweep(s_cur, v_cur):

    # # OPTION 1: return "earliest" set
    # s_trans = Sphere(s_cur.pos, s_cur.rad)
    # # translate s_trans along v_cur
    # while np.linalg.norm(s_trans.pos - s_cur.pos) < np.linalg.norm(v_cur):
    #     inter = tree.query_ball_point(s_trans.pos, r + r)
    #     if len(inter) > 0:
    #         return set(spheres[i] for i in inter)
    #     s_trans.pos += r * v_cur / np.linalg.norm(v_cur) # iterate by distance r
    # return set()

    # OPTION 2: return entire set
    inter = set()
    s_trans = Sphere(s_cur.pos, s_cur.rad)

    # translate s_trans along v_cur
    while np.linalg.norm(s_trans.pos - s_cur.pos) < np.linalg.norm(v_cur):
        query = tree.query_ball_point(s_trans.pos, r + r)
        inter.update(set(spheres[i] for i in query))
        s_trans.pos = s_trans.pos + r * v_cur / np.linalg.norm(v_cur) # iterate by distance r

    s_trans.pos = s_cur.pos + v_cur # iterate to end of vector
    query = tree.query_ball_point(s_trans.pos, r + r)
    inter.update(set(spheres[i] for i in query))
    inter.remove(s_cur)
    return inter

# even if s_cur sweeps all the way past some sphere, 
# that sphere still has to move "past" s_cur for its DV

def simulate(s1, s2):
    outstanding = []
    q = Queue()
    v1 = compute_dv(s1, s2)
    v2 = compute_dv(s2, s1)
    q.put((s1, v1))
    q.put((s2, v2))
    while not q.empty():
        s_cur, v_cur = q.get()
        outstanding.append((s_cur, v_cur))
        for s_new in sweep(s_cur, v_cur):
            if (s_cur == s1 and s_new == s2) or (s_cur == s2 and s_new == s1): # prevent infinite loop
                continue
            s_trans = Sphere(s_cur.pos + v_cur, s_cur.rad)
            v_new = compute_dv(s_new, s_trans)
            q.put((s_new, v_new))
    return outstanding


if __name__ == '__main__':

    # https://www.geogebra.org/3d/ma7vpx3m
    points = [
        np.array([2.5, -0.5, 2.5]),
        np.array([0.0, 2.0, 1.0]),
        np.array([-1.0, 0.0, 4.0]),
        np.array([1.0, -2.0, 2.0]),
        np.array([1.0, -1.0, 0.0]),
        np.array([0.5, -3.0, 1.5]),
        np.array([-1.0, -3.0, 0.0])
    ]

    tree = KDTree(points)
    for p in points:
        spheres.append(Sphere(p, r))

    inter = tree.query_pairs(r + r).pop()
    outstanding = simulate(spheres[inter[0]], spheres[inter[1]])
    for s, v in outstanding:
        print(f'({spheres.index(s) + 1}) {s.pos} : {v}')

