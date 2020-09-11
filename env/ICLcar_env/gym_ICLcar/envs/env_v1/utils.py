import math
import numpy as np

def wrap2pi(rad):
    while rad > 2*math.pi:
        rad -= 2*math.pi

    while rad < 0:
        rad += 2*math.pi

    return rad

def wrap(list, indx, offset):
    return ((indx + offset) + len(list)) % len(list)

def vec2angle(vec):
    x, y = vec
    rad = math.atan2(y,x)
    rad = wrap2pi(rad)
    angle = rad * 180/math.pi
    return angle
