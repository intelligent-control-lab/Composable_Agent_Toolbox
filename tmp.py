"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math

# Initializing number of dots
N = 25

# Creating dot class
class dot(object):
    def __init__(self):
        self.x = 10 * np.random.random_sample()
        self.y = 10 * np.random.random_sample()
        self.velx = self.generate_new_vel()
        self.vely = self.generate_new_vel()

    def generate_new_vel(self):
        return (np.random.random_sample() - 0.5) / 5

    def move(self):
        def distance(x1, y1, x2, y2):
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        def inside(x1, y1):
            if distance(x1, y1, 5, 5) <= 1:
                return True
            else:
                return False

        def calc_dist(d):
            ret = 0
            for x in dots:
                if inside(x.x, x.y) and x != d:                            
                    ret = ret + distance(x.x, x.y, d.x, d.y)
            return ret

        # if dot is inside the circle it tries to maximize the distances to
        # other dots inside circle
        if inside(self.x, self.y):
            dist = calc_dist(self)
            for i in range(1, 10):
                self.velx = self.generate_new_vel()
                self.vely = self.generate_new_vel()
                self.x = self.x + self.velx
                self.y = self.y + self.vely
                if calc_dist(self) <= dist or not inside(self.x, self.y):
                    self.x = self.x - self.velx
                    self.y = self.y - self.vely
        else:
            if np.random.random_sample() < 0.95:
                self.x = self.x + self.velx
                self.y = self.y + self.vely
            else:
                self.velx = self.generate_new_vel()
                self.vely = self.generate_new_vel()
                self.x = self.x + self.velx
                self.y = self.y + self.vely
            if self.x >= 10:
                self.x = 10
                self.velx = -1 * self.velx
            if self.x <= 0:
                self.x = 0
                self.velx = -1 * self.velx
            if self.y >= 10:
                self.y = 10
                self.vely = -1 * self.vely
            if self.y <= 0:
                self.y = 0
                self.vely = -1 * self.vely

# Initializing dots
dots = [dot() for i in range(N)]

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
d, = ax.plot([dot.x for dot in dots],
             [dot.y for dot in dots], 'ro')
circle = plt.Circle((5, 5), 1, color='b', fill=False)
ax.add_artist(circle)


# animation function.  This is called sequentially
def animate(i):
    for dot in dots:
        dot.move()
    d.set_data([dot.x for dot in dots],
               [dot.y for dot in dots])
    return d,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=200, interval=20)

plt.show()