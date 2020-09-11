# the testing files for random track generalization
import numpy as np
from math import pi
from ipdb import set_trace
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class cpoint():
  def __init__(self, theta, dist, max_distance=500, min_distance=250):
    self.theta = theta
    self.dist = dist
    self.rcp = None # right control point
    self.lcp = None # left control point
    self.max_distance = max_distance
    self.min_distance = min_distance
    self.auxiliary_cp() # set the auxiliary left and right control point

  @property
  def x(self): return self.dist*np.cos(self.theta)

  @property
  def y(self): return self.dist*np.sin(self.theta)

  def auxiliary_cp(self):
    '''
    input: the control points sequence
    '''
    length = np.random.rand((1)) * (self.max_distance - self.min_distance) + self.min_distance # the maximum distance of auxiliary cpoints
    theta = np.random.rand((1)) * 2 * pi

    pos1 = np.append(np.array([self.x + length*np.cos(theta), self.y + length*np.sin(theta)]),0)
    pos2 = np.append(np.array([self.x - length*np.cos(theta), self.y - length*np.sin(theta)]),0)
    # judge the left or right points
    # use cross product
    pos = np.array([self.x, self.y, 0])
    vec1 = pos1 - np.array([self.x, self.y, 0])
    vec2 = pos2 - np.array([self.x, self.y, 0])
    cross1 = np.cross(pos1, vec1)
    cross2 = np.cross(pos2, vec2)
    if cross1[2] > 0:
      self.lcp = pos1[:2]
      self.rcp = pos2[:2]
    else:
      self.lcp = pos2[:2]
      self.rcp = pos1[:2]


def generate_track():
  dist_max = 1500
  dist_min = 600
  dist_interval = dist_max - dist_min
  cpoint_num = 10

  # random sampling the control points
  # sample the radius
  # radius = np.sort(np.random.rand(cpoint_num,1) * 2 * pi, axis=0)
  radius = np.expand_dims(np.linspace(0, (cpoint_num-1)/cpoint_num*2*pi, cpoint_num), axis=1)
  # sample the distance
  dist = np.random.rand(cpoint_num,1) * dist_interval + dist_min

  # create the corresonding control point
  info = np.hstack((radius, dist))
  cpoints = []
  xlist = []
  ylist = []
  for i in range(cpoint_num):
    cp = cpoint(info[i,0], info[i,1])
    xlist.append(cp.x)
    ylist.append(cp.y)
    cpoints.append(cp)
  center_lane, comp, center = b_curve_fitting(cpoints)
  # # visuliazation for the points
  # plt.clf()
  # # figure = Figure()
  # # canvas = FigureCanvas(figure)
  # # axes = figure.add_subplot(1, 1, 1, axisbg='red')
  # # axes.plot(center_lane[:,0], center_lane[:,1], '-', linewidth=2, markersize=12)
  # # plt.plot(center_lane[:,0], center_lane[:,1], 'go', linewidth=1, markersize=12, color='gray')
  # plt.scatter(center_lane[::4,0], center_lane[::4,1], s=2)
  # # plt.plot(comp[:,0], comp[:,1], '-', linewidth=2, markersize=12, alpha=0.4)
  # # plt.plot(center[:,0], center[:,1], 'go', linewidth=2, markersize=4)
  # ax = plt.gca()
  # ax.set_facecolor('black')
  # # ax.set_facecolor((1.0, 0.47, 0.42))
  # plt.pause(2)
  center_lane += 1500
  return center_lane, comp, center


def b_curve_fitting(cpoints):
  center_lane = np.zeros(0)
  comp = np.zeros(0)
  center = np.zeros(0)
  for i in range(len(cpoints)):
  # for i in range(2):
    P0 = np.array([cpoints[i].x, cpoints[i].y])
    P1 = cpoints[i].lcp
    P2 = cpoints[(i+1)%len(cpoints)].rcp
    P3 = np.array([cpoints[(i+1)%len(cpoints)].x, cpoints[(i+1)%len(cpoints)].y])
    # get the sequence of center line points
    ts = np.linspace(0,1,100)
    for t in ts:
      p_tmp = (1-t)**3*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3
      if center_lane.shape[0] == 0:
        center_lane = p_tmp
      else:
        center_lane = np.vstack((center_lane, p_tmp))
    # record comp
    if comp.shape[0] == 0:
      comp = P0
      comp = np.vstack((comp, P1))
      comp = np.vstack((comp, P2))
      comp = np.vstack((comp, P3))
    else:
      comp = np.vstack((comp, P0))
      comp = np.vstack((comp, P1))
      comp = np.vstack((comp, P2))
      comp = np.vstack((comp, P3))

    # record center point
    if center.shape[0] == 0:
      center = P0
    else:
      center = np.vstack((center, P0))

  return center_lane, comp, center


if __name__ == "__main__":
  generate_track()