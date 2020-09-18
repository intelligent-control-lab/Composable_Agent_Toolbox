import numpy as np
import os
import yaml
import re

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