import numpy as np
import os
import yaml
import re
from numpy import cos, sin
from math import inf
from ipdb import set_trace


class cap():
    def __init__(self):
        # the class for capsule object
        self.p = np.zeros((3,2))
        self.r = 0


def jac_num(ineq, x, obs_p, eps=1e-6):
    '''
    compoute the jaccobian for a given function 
    used for computing first-order gradient of distance function 
    '''
    # y = ineq(x,obs_p)

    # # change to unified n-d array format
    # if type(y) == np.float64:
    #     y = np.array([y])

    # grad = np.zeros((y.shape[0], x.shape[0]))
    # xp = x
    # for i in range(x.shape[0]):
    #     xp[i] = x[i] + eps/2
    #     yhi = ineq(xp,obs_p)
    #     xp[i] = x[i] - eps/2
    #     ylo = ineq(xp,obs_p)
    #     grad[:,i] = (yhi - ylo) / eps
    #     xp[i] = x[i]
    # return grad

    # use the analytical solution 
    obs_p = obs_p.flatten()
    
    # flatten the input x 
    x = x.flatten()
    dist = np.linalg.norm(x - obs_p)
    grad = np.zeros((1, 2))
    grad[:,0] = 0.5/dist*2*(x[0]-obs_p[0])
    grad[:,1] = 0.5/dist*2*(x[1]-obs_p[1])
    return grad


def jac_num_arm(ineq, x,DH,base,obs,cap, eps=1e-6):
    '''
    compoute the jaccobian for a given function 
    used for computing first-order gradient of distance function 
    '''
    y = ineq(x,DH,base,obs,cap)

    # change to unified n-d array format
    if type(y) == np.float64:
        y = np.array([y])

    grad = np.zeros((y.shape[0], x.shape[0]))
    xp = x
    for i in range(x.shape[0]):
        xp[i] = x[i] + eps/2
        yhi = ineq(xp,DH,base,obs,cap)
        xp[i] = x[i] - eps/2
        ylo = ineq(xp,DH,base,obs,cap)
        grad[:,i] = (yhi - ylo) / eps
        xp[i] = x[i]
    return grad

    # use the analytical solution 
    # obs_p = obs_p.flatten()
    
    # # flatten the input x 
    # x = x.flatten()
    # dist = np.linalg.norm(x - obs_p)
    # grad = np.zeros((1, 2))
    # grad[:,0] = 0.5/dist*2*(x[0]-obs_p[0])
    # grad[:,1] = 0.5/dist*2*(x[1]-obs_p[1])
    # return grad


def load_experiment_settings(experiment_settings, filepath=None):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    all_settings = {}
    if experiment_settings and experiment_settings[0]:
        for experiment_setting in experiment_settings:
            with open(experiment_setting, 'r') as f:
                settings = yaml.load(f, Loader=loader)
                if settings:
                    all_settings.update(settings)
    return all_settings




from numpy import cos, sin


def set_reference_traj(robot):

    """ define the steps and other variables. """
    tfinal = 500
    nstep = 100
    nwait = 30
    r = 0.25
    v = 25
    pos_ini = robot.pos[0]
    pos_exe = np.array([1.5, 0.0010, 1.0]).reshape((3, 1))

    """ set the reference trajectory, goal(1:3) """
    diff = pos_exe - pos_ini
    for t in range(nstep):
        robot.goal.append(pos_ini + (t / nstep) * diff)

    for t in range(nwait):
        robot.goal.append(pos_exe)

    for t in range(tfinal):
        robot.goal.append(pos_exe + r * np.array([0, sin(t / v), 0]).reshape((3, 1)))

    """ set the reference velocity """
    for t in range(nstep + nwait + tfinal - 1):
        vel_tmp = (robot.goal[t+1][0:3, 0:1] - robot.goal[t][0:3, 0:1]) / robot.delta_t
        robot.goal[t] = np.block([[robot.goal[t]],
                                  [vel_tmp]])
    # the last step, velocity is 0 vector
    robot.goal[nstep + nwait + tfinal - 1] = np.block([[robot.goal[nstep + nwait + tfinal - 1]],
                                                       [np.zeros((3, 1))]])

    """ set the reference acceration """
    for t in range(nstep + nwait + tfinal - 2):
        acc_tmp = (robot.goal[t + 1][3:6, 0:1] - robot.goal[t][3:6, 0:1]) / robot.delta_t
        robot.goal[t] = np.block([[robot.goal[t]],
                                  [acc_tmp]])

    # the last two steps, velocity is zero vector
    robot.goal[nstep + nwait + tfinal - 1] = np.block([[robot.goal[nstep + nwait + tfinal - 1]],
                                                       [np.zeros((3, 1))]])
    robot.goal[nstep + nwait + tfinal - 2] = np.block([[robot.goal[nstep + nwait + tfinal - 2]],
                                                       [np.zeros((3, 1))]])
    """ return robot property and step setting """
    return [robot, tfinal, nstep, nwait]



def jacobi_diff(DH, n, p, qdot):
    """
    :param DH: DH parameter of robot
    :param n: number of degree of freedom (nlink)
    :param p: end-effector position
    :param qdot: velocity of joints
    :return: Jacobi matrix (J) and its derivative (H)

    This function This function calculates the Jacobi matrix (J)
    and its derivative (H) of a given point p (in the base
    coordinate) on the link n.
    """
    z = np.empty((3, n))
    r_0 = np.empty((3, n+1))
    w = np.empty((3, n))
    J = np.empty((6, n))
    vj = np.empty((3, n))

    # initialize the above variable to be nan
    z[:] = np.nan
    r_0[:] = np.nan
    w[:] = np.nan
    J[:] = np.nan
    vj[:] = np.nan

    # other variable
    TCP_T = np.identity(4)
    JointJacobi = []

    # initialize the jointJac
    for i in range(1, n+1):
        tmp = np.empty((3, i))
        tmp[:] = np.nan
        JointJacobi.append(tmp)

    # get DH parameters
    alpha = DH[:, 3:]
    A = DH[:, 2:3]
    D = DH[:, 1:2]
    q = DH[:, 0:1]

    for i in range(n):
        z[:, i:i+1] = TCP_T[0:3, 2:3]
        r_0[:, i:i+1] = TCP_T[0:3, 3:4]
        if i > 0:
            w[:, i:i+1] = w[:, i-1:i] + z[:, i:i+1] * qdot[i]
        else:
            w[:, i:i+1] = z[:, i:i+1] * qdot[i]

        tmp_M = np.array([[cos(q[i, 0]), -sin(q[i, 0]) * cos(alpha[i, 0]), sin(q[i, 0]) * sin(alpha[i, 0]), A[i, 0] * cos(q[i, 0])],
                          [sin(q[i, 0]), cos(q[i, 0]) * cos(alpha[i, 0]), -cos(q[i, 0]) * sin(alpha[i, 0]), A[i, 0] * sin(q[i, 0])],
                          [0,         sin(alpha[i, 0]),             cos(alpha[i, 0]),              D[i, 0]],
                          [0,         0,                         0,                          1]])
        TCP_T = np.matmul(TCP_T, tmp_M)

    r_0[:, n:] = TCP_T[0:3, 3:4]

    for i in range(n):
        J[:, i:i+1] = np.block([[np.cross(r_0[:, i:i+1] - p, z[:, i:i+1], axis=0)],
                                [z[:, i:i+1]]])

    return J


def cap_pos(base, DH, cap_end):
    nlink = DH.shape[0]
    M = []
    M.append(np.identity(4))

    for i in range(nlink):
        R = np.array([[cos(DH[i][0]), -sin(DH[i][0]) * cos(DH[i][3]), sin(DH[i][0]) * sin(DH[i][3])],
                      [sin(DH[i][0]), cos(DH[i][0]) * cos(DH[i][3]), -cos(DH[i][0])*sin(DH[i][3])],
                      [0, sin(DH[i][3]), cos(DH[i][3])]])

        T = np.array([[DH[i][2] * cos(DH[i][0])],
                      [DH[i][2] * sin(DH[i][0])],
                      [DH[i][1]]])

        RT = np.block([[R,                T],
                       [np.zeros((1, 3)), 1]])

        M_tmp = np.matmul(M[i], RT)
        M.append(M_tmp)

    epos = np.matmul(M[nlink][0:3, 0:3], cap_end)
    epos = epos + M[nlink][0:3, 3:] + base
    return epos


def cap_pos_ori(base, DH, rob_cap):
    '''
    the complete cappos computation tools 
    '''
    nlink = DH.shape[0]
    M = []
    pos = []
    M.append(np.identity(4))

    for i in range(nlink):
        R = np.array([[cos(DH[i][0]), -sin(DH[i][0]) * cos(DH[i][3]), sin(DH[i][0]) * sin(DH[i][3])],
                      [sin(DH[i][0]), cos(DH[i][0]) * cos(DH[i][3]), -cos(DH[i][0])*sin(DH[i][3])],
                      [0, sin(DH[i][3]), cos(DH[i][3])]])

        T = np.array([[DH[i][2] * cos(DH[i][0])],
                      [DH[i][2] * sin(DH[i][0])],
                      [DH[i][1]]])

        RT = np.block([[R,                T],
                       [np.zeros((1, 3)), 1]])

        M_tmp = np.matmul(M[i], RT)
        M.append(M_tmp)
        # get a new cappos
        tmp_cap = cap()
        for j in range(2):
            tmp_cap.p[:,j] = np.matmul(M_tmp[0:3,0:3], rob_cap[i].p[:,j]) + M_tmp[0:3,3] + np.squeeze(base)
            tmp_cap.r = rob_cap[i].r
        pos.append(tmp_cap)

    return pos


def distLinSeg(point1s, point1e, point2s, point2e):
    '''
    Function for fast computation of the shortest distance between two line segments
    '''
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = np.sum(np.power(d1,2))
    D2 = np.sum(np.power(d2,2))

    S1 = np.sum(d1*d12)
    S2 = np.sum(d2*d12)
    R = np.sum(d1*d2)

    den = D1*D2 - R**2

    if D1 == 0 or D2 == 0:
        if D1 != 0:
            u = 0
            t = S1/D1
            t = fixbound(t)
        elif D2 != 0:
            t = 0
            u = -S2/D2
            U = fixbound(u)
        else:
            t = 0
            u = 0
    elif den == 0:
        t = 0
        u = -S2/D2
        uf = fixbound(u)
        if uf != u:
            t = (uf*R+S1)/D1
            t = fixbound(t)
            u = uf 
    else:
        t = (S1*D2-S2*R)/den
        t = fixbound(t)
        u = (t*R-S2)/D2
        uf = fixbound(u)
        if uf != u:
            t = (uf*R+S1)/D1
            t = fixbound(t)
            u = uf
    # compute distance given parameters t and u 
    dist = np.linalg.norm(d1*t - d2*u - d12)
    # dist = np.sqrt(np.sum(np.power(d1*t-d2*u-d12,2)))
    # compute the cloest point 
    points = np.vstack((point1s + d1*t, point2s+d2*u)).transpose()
    return dist, points




def fixbound(num):
    '''
    if num is out of (0,1) round to {0,1}
    '''
    if num < 0:
        num = 0
    elif num > 1:
        num = 1
    return num



def distance_arm(theta,DH,base,obs,cap):
    '''
    3d distance function for robot arm 
    '''

    nstate = DH.shape[0]
    for i in range(nstate):
        DH[i,0] = theta[i]
    d = inf
    pos = cap_pos_ori(base, DH, cap)

    for i in range(nstate):
        if obs.shape[1] == 2:
            
            dis, points = distLinSeg(pos[i].p[:,0],pos[i].p[:,1], obs[:,0],obs[:,1])
        elif obs.shape[1] == 1:
            dis, points = distLinSeg(pos[i].p[:,0],pos[i].p[:,1], obs[:,0],obs[:,0])
        if np.linalg.norm(dis)<0.0001:
            dis = -np.linalg.norm(points[:,1]-pos[i].p[:,2])
         
        if dis < d:
            d = dis
            linkid=i
    return d


def vstack_wrapper(a, b):
    if a == []:
        return b
    else:
        x = np.vstack((a,b))
        return x

def mat_like(M):
    '''
    make sure the M is 2d array
    '''
    pass