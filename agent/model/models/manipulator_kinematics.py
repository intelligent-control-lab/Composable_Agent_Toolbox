#
from abc import ABC, abstractmethod  # This is needed for the abstract classes

# Third-party Imports
import numpy as np
import sympy as sp
import dill

# Application Specific Imports
from agent.model.models.models import ModelBase

class ManipulatorKinematicsModel(ABC):
	def __init__(self, spec):
        '''
        class constructor for a nonlinear model class.

        This class constructor takes a spec that specifies the following options:
            use_spec    - exp: boolean - whether to leverage a spec defintion
            use_library - exp: boolean - whether to leverage dynamics library
            model_name  - exp: string - name of the model (in dynamics library)
            time_sample - exp: double/float - time sample value
            disc_flag   - exp: boolean - discrete time flag modifier
        '''
        
        self.nlink = spec['nlink']
        self.DH = spec['DH']

        self.lb = spec['lb'];
        self.ub = spec['ub'];

        self.base = spec['base'] # base link offset
        self.cap_end = spec['cap_end']

        # robot capsule for each link. 
        # Current capsule is not specified. 
        # End-effector measured offset, with respect to last coorfinate
        self.cap = []
        self.define_capsule()



	def cap_pos(self, base, DH, cap_end):
		'''
		Forward kinematics 
		'''
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

	
	def jacobi_diff(self, DH, n, p, qdot):
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


	def define_capsule(self):
        for i in range(self.nlink):
            tmp = cap()
            self.cap.append(tmp)
        # set the capsules
        self.cap[0].p = np.array([[-0.145, -0.145], [0.105, 0.105], [0, 0]])
        self.cap[0].r = 0.385
        
        self.cap[1].p = np.array([[-0.87, 0], [0, 0], [-0.1945, -0.1945]])
        self.cap[1].r = 0.195
        
        self.cap[2].p = np.array([[-0.02, -0.09], [0.073, 0.073], [0.115, 0.115]])
        self.cap[2].r = 0.33
        
        self.cap[3].p = np.array([[0, 0], [-0.65, 0], [-0.0235, -0.0235]])
        self.cap[3].r = 0.115
        
        self.cap[4].p = np.array([[0, 0], [0.0145, 0.0145], [0.025, 0.025]])
        self.cap[4].r = 0.15

        self.cap[5].p = np.array([[0.00252978728478826, 0.000390496336481267], [6.28378116607958e-10,1.00300828261106e-10], [-0.170767309373314,-0.0344384157898974]])
        self.cap[5].r = 0.03