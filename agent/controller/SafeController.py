import sys, os
from os.path import abspath, join, dirname
from env import safety_gym_env
sys.path.insert(0, join(abspath(dirname(__file__)), '../../'))

import numpy as np
from abc import ABC, abstractmethod
from utils import GoalType
import cvxopt
import copy

class SafeController(ABC):
    def __init__(self, spec, model):

        self.model = model
        self.u_max = np.vstack(spec["u_max"])

    @abstractmethod
    def __call__(self,
        dt: float,
        processed_data: dict,
        u_ref: np.ndarray,
        goal: np.ndarray,
        goal_type: GoalType) -> np.ndarray:
        '''
            Driver procedure. Do not change
        '''
        pass

# TODO ISSA: new child class of SafeController
class ISSAController(SafeController):
    def __call__(self,
        dt: float,
        processed_data: dict,
        u_ref: np.ndarray,
        goal: np.ndarray,
        goal_type: GoalType) -> np.ndarray:
        '''
            Driver procedure. Do not change
        '''
        
        u_new, valid_adamba_sc, processed_data['safety_gym_env'].safety_gym_env, all_satisfied_u = self.adamba_safecontrol(processed_data['state'], [u_ref], env=processed_data['safety_gym_env'].safety_gym_env)
        return u_new, None

    def adamba_safecontrol(self, s, u, env, threshold=0, dt_ratio=1.0, ctrlrange=10.0, margin=0.4, adaptive_k=3, adaptive_n=1, adaptive_sigma=0.04, trigger_by_pre_execute=False, pre_execute_coef=0.0, vec_num=None, max_trial_num =1):

        infSet = []

        u = np.clip(u, -ctrlrange, ctrlrange)

        action_space_num = env.action_space.shape[0]
        action = np.array(u).reshape(-1, action_space_num)

        

        dt_adamba = env.model.opt.timestep * env.frameskip_binom_n * dt_ratio

        assert dt_ratio == 1

        limits= [[-ctrlrange, ctrlrange]] * action_space_num
        NP = action

        # generate direction
        NP_vec_dir = []
        NP_vec = []
    
        #sigma_vec = [[1, 0], [0, 1]]
        # vec_num = 10 if action_space_num==2 else 
        if action_space_num ==2:
            vec_num = 10 if vec_num == None else vec_num
        elif action_space_num == 12:
            vec_num = 20 if vec_num == None else vec_num
        else:
            raise NotImplementedError

        loc = 0 
        scale = 0.1
        
        # num of actions input, default as 1
        for t in range(0, NP.shape[0]):
            if action_space_num == 2:
                vec_set = []
                vec_dir_set = []
                for m in range(0, vec_num):
                    # vec_dir = np.random.multivariate_normal(mean=[0, 0], cov=sigma_vec)
                    theta_m = m * (2 * np.pi / vec_num)
                    vec_dir = np.array([np.sin(theta_m), np.cos(theta_m)]) / 2
                    #vec_dir = vec_dir / np.linalg.norm(vec_dir)
                    vec_dir_set.append(vec_dir)
                    vec = NP[t]
                    vec_set.append(vec)
                NP_vec_dir.append(vec_dir_set)
                NP_vec.append(vec_set)
            else:
                vec_dir_set = np.random.normal(loc=loc, scale=scale, size=[vec_num, action_space_num])
                vec_set = [NP[t]] * vec_num
                #import ipdb; ipdb.set_trace()
                NP_vec_dir.append(vec_dir_set)
                NP_vec.append(vec_set)
    
        bound = 0.0001
        # record how many boundary points have been found
        # collected_num = 0
        valid = 0
        cnt = 0
        out = 0
        yes = 0
        
        max_trials = max_trial_num
        for n in range(0, NP.shape[0]):
            trial_num = 0
            at_least_1 = False
            while trial_num < max_trials and not at_least_1:
                at_least_1 = False
                trial_num += 1
                NP_vec_tmp = copy.deepcopy(NP_vec[n])
                #print(NP_vec)

                if trial_num ==1:
                    NP_vec_dir_tmp = NP_vec_dir[n]
                else:
                    NP_vec_dir_tmp = np.random.normal(loc=loc, scale=scale, size=[vec_num, action_space_num])


                for v in range(0, vec_num):
                    NP_vec_tmp_i = NP_vec_tmp[v]

                    NP_vec_dir_tmp_i = NP_vec_dir_tmp[v]

                    eta = bound
                    decrease_flag = 0
                    # print(eta)
                    
                    while True: 
                        #print(eta)
                        # chk_start_time = time.time()
                        flag, env = self.chk_unsafe(s, NP_vec_tmp_i, dt_ratio=dt_ratio, dt_adamba=dt_adamba, env=env,
                                            threshold=threshold, margin=margin, adaptive_k=adaptive_k, adaptive_n=adaptive_n, adaptive_sigma=adaptive_sigma,
                                            trigger_by_pre_execute=trigger_by_pre_execute, pre_execute_coef=pre_execute_coef)

                        # safety gym env itself has clip operation inside
                        if self.outofbound(limits, NP_vec_tmp_i):
                            break

                        if eta <= bound and flag==0:
                            at_least_1 = True
                            break

                        # AdamBA procudure
                        if flag == 1 and decrease_flag == 0:
                            # outreach
                            NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                            eta = eta * 2
                            continue
                        # monitor for 1st reaching out boundary
                        if flag == 0 and decrease_flag == 0:
                            decrease_flag = 1
                            eta = eta * 0.25  # make sure decrease step start at 0.5 of last increasing step
                            continue
                        # decrease eta
                        if flag == 1 and decrease_flag == 1:
                            NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                            eta = eta * 0.5
                            continue
                        if flag == 0 and decrease_flag == 1:
                            NP_vec_tmp_i = NP_vec_tmp_i - eta * NP_vec_dir_tmp_i
                            eta = eta * 0.5
                            continue

                    NP_vec_tmp[v] = NP_vec_tmp_i

            NP_vec_tmp_new = []
            # print("NP_vec_tmp: ",NP_vec_tmp)
            # exit(0)

            # print(u)
            # print(NP_vec_tmp)
            for vnum in range(0, len(NP_vec_tmp)):
                cnt += 1
                if self.outofbound(limits, NP_vec_tmp[vnum]):
                    # print("out")
                    out += 1
                    continue
                if NP_vec_tmp[vnum][0] == u[0][0] and NP_vec_tmp[vnum][1] == u[0][1]:
                    # print("yes")
                    yes += 1
                    continue

                valid += 1
                NP_vec_tmp_new.append(NP_vec_tmp[vnum])
            NP_vec[n] = NP_vec_tmp_new

        NP_vec_tmp = NP_vec[0]

        if valid > 0:
            valid_adamba_sc = "adamba_sc success"
        elif valid == 0 and yes==vec_num:
            valid_adamba_sc = "itself satisfy"
        elif valid == 0 and out==vec_num:
            valid_adamba_sc = "all out"
        else:
            valid_adamba_sc = "exception"
            print("out = ", out)
            print("yes = ", yes)
            print("valid = ", valid)

        if len(NP_vec_tmp) > 0:  # at least we have one sampled action satisfying the safety index 
            norm_list = np.linalg.norm(NP_vec_tmp, axis=1)
            optimal_action_index = np.where(norm_list == np.amin(norm_list))[0][0]
            return NP_vec_tmp[optimal_action_index], valid_adamba_sc, env, NP_vec_tmp
        elif valid_adamba_sc == 'itself satisfy':
            return u, valid_adamba_sc, env, None
        else:
            return None, valid_adamba_sc, env, None
        
    def chk_unsafe(self, s, point, dt_ratio, dt_adamba, env, threshold, margin, adaptive_k, adaptive_n, adaptive_sigma, trigger_by_pre_execute, pre_execute_coef):
        action = point.tolist()
        # save state of env
        stored_state = copy.deepcopy(env.sim.get_state())
        safe_index_now = env.adaptive_safety_index(k=adaptive_k, sigma=adaptive_sigma, n=adaptive_n)

        # simulate the action
        s_new = env.step(action, simulate_in_adamba=True)

        safe_index_future = env.adaptive_safety_index(k=adaptive_k, sigma=adaptive_sigma, n=adaptive_n)

        dphi = safe_index_future - safe_index_now

        if trigger_by_pre_execute:
            if safe_index_future < pre_execute_coef:
                flag = 0  # safe
            else:
                flag = 1  # unsafe
        else:
            if dphi <= threshold * dt_adamba: #here dt_adamba = dt_env
                flag = 0  # safe
            else:
                flag = 1  # unsafe

        # set qpos and qvel
        env.sim.set_state(stored_state)
        
        # Note that the position-dependent stages of the computation must have been executed for the current state in order for these functions to return correct results. So to be safe, do mj_forward and then mj_jac. If you do mj_step and then call mj_jac, the Jacobians will correspond to the state before the integration of positions and velocities took place.
        env.sim.forward()
        return flag, env

    def outofbound(self, action_limit, action):
        flag = False
        for i in range(len(action_limit)):
            assert action_limit[i][1] > action_limit[i][0]
            if action[i] < action_limit[i][0] or action[i] > action_limit[i][1]:
                flag = True
                break
        return flag

class UnsafeController(SafeController):
    def __call__(self,
        dt: float,
        processed_data: dict,
        u_ref: np.ndarray,
        goal: np.ndarray,
        goal_type: GoalType) -> np.ndarray:
        '''
            Driver procedure. Do not change
        '''
        return u_ref, None




class EnergyFunctionController(SafeController):
    """
    Energy function based safe controllers

    Attributes:
        _name
        _spec
        _model
    """
    def __init__(self, spec, model):
        self._spec = spec
        self._model = model
        self.d_min = spec['d_min']
        self.eta = spec['eta']
        self.k_v = spec['k_v']

        self.prev_phi = None

    def phi_and_derivatives(self, dt, ce, co):
        """
        ce: cartesian position of ego
        co: cartesian position of an obstacle
        """
        n = np.shape(ce)[0]//2

        dp = np.vstack(ce[:n] - co[:n])
        dv = np.vstack(ce[n:] - co[n:])

        d     = max(np.linalg.norm(dp), 1e-3)
        dot_d = dp.T @ dv / d

        phi = self.d_min**2 - d**2 - self.k_v * dot_d + self.eta * dt

        p_phi_p_d = -2 * d
        p_phi_p_dot_d = - self.k_v

        p_d_p_ce = np.vstack([dp / d, np.zeros((n,1))])
        p_d_p_co = -p_d_p_ce

        p_dot_d_p_dp = dv / d - np.asscalar(dp.T @ dv) * dp / (d**3)
        p_dot_d_p_dv = dp / d

        p_dp_p_ce = np.hstack([np.eye(n), np.zeros((n,n))])
        p_dp_p_co = -p_dp_p_ce

        p_dv_p_ce = np.hstack([np.zeros((n,n)), np.eye(n)])
        p_dv_p_co = -p_dv_p_ce

        p_dot_d_p_ce = p_dp_p_ce.T @ p_dot_d_p_dp + p_dv_p_ce.T @ p_dot_d_p_dv
        p_dot_d_p_co = p_dp_p_co.T @ p_dot_d_p_dp + p_dv_p_co.T @ p_dot_d_p_dv

        p_phi_p_ce = p_phi_p_d * p_d_p_ce + p_phi_p_dot_d * p_dot_d_p_ce
        p_phi_p_co = p_phi_p_d * p_d_p_co + p_phi_p_dot_d * p_dot_d_p_co
        
        return phi, p_phi_p_ce, p_phi_p_co
    
    @abstractmethod
    def safe_control(self, u_ref, obs, dt, processed_data):
        """ Compute the safe control between ego and an obstacle.
        """
        pass
    
    def __call__(self,
        dt: float,
        processed_data: dict,
        u_ref: np.ndarray,
        goal: np.ndarray,
        goal_type: GoalType) -> np.ndarray:
        '''
            Driver procedure. Do not change
        '''
        us = []

        for obs in processed_data["obstacle_sensor_est"]:
            phi, u = self.safe_control(u_ref, obs, dt, processed_data)
            us.append((phi, u))
        sorted(us, key=lambda x:x[0], reverse=True) # larger phi first
        
        phi = us[0][0]

        if self.prev_phi is None or phi <= 0:
            dphi = 0
        else:
            dphi = phi - self.prev_phi

        self.prev_phi = phi

        return us[0][1], dphi # adopt the control that avoids the most dangerous collision.



class SafeSetController(EnergyFunctionController):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self._name = 'safe_set'

    def safe_control(self, u_ref, obs, dt, processed_data):
        """ Compute the safe control between ego and an obstacle.

        Safe set compute u by solving the following optimization:
        min || u - u_ref ||, 
        s.t.  dot_phi < eta  or  phi > 0 (eta is the safety margin used in phi)

        => p_phi_p_xe.T * dot_xe        + p_phi_p_co.T * dot_co < eta
        => p_phi_p_xe.T * (fx + fu * u) + p_phi_p_co.T * dot_co < eta
        => p_phi_p_xe.T * fu * u < eta - p_phi_p_xe.T * fx - p_phi_p_co.T * dot_co

        """
        ce = np.vstack([processed_data["cartesian_sensor_est"]["pos"], processed_data["cartesian_sensor_est"]["vel"]])  # ce: cartesian state of ego
        co = np.vstack([processed_data["obstacle_sensor_est"][obs]["rel_pos"], processed_data["obstacle_sensor_est"][obs]["rel_vel"]]) + ce  # co: cartesian state of the obstacle
        
        x =  np.vstack(processed_data["state_sensor_est"]["state"])

        n = np.shape(ce)[0]//2

        # It will be better if we have an estimation of the acceleration of the obstacle
        dot_co = np.vstack([co[n:], np.zeros((n,1))])

        phi, p_phi_p_ce, p_phi_p_co = self.phi_and_derivatives(dt, ce, co)

        p_ce_p_xe = self._model.jacobian(x)
        # dot_x = fx + fu * u
        fx = self._model.fx(x)
        fu = self._model.fu(x)

        p_phi_p_xe = p_ce_p_xe.T @ p_phi_p_ce

        L = p_phi_p_xe.T @ fu
        S = -self.eta - p_phi_p_xe.T @ fx - p_phi_p_co.T @ dot_co
        
        u = u_ref

        phi = phi[0][0]

        if phi <= 0 or np.asscalar(L @ u_ref) < np.asscalar(S):
            u = u_ref
        else:
            u = u_ref - (np.asscalar(L @ u_ref - S) * L.T / np.asscalar(L @ L.T))

        return phi, u


class PotentialFieldController(EnergyFunctionController):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self._name = 'potential_field'
        self.c = spec['c']
        
    def safe_control(self, u_ref, obs, dt, processed_data):
        """ Compute the safe control between ego and an obstacle.

        Potential Field compute the safe control by first computing a safe control in the cartesian space,
        then compute the state space control by inverse kinematics.

        This method is not suitable for all control systems.
        
        u_cartesian = -c * p_phi_p_ce,
        u = fu.T * p_ce_p_xe.T * u_ce

        """
        ce = np.vstack([processed_data["cartesian_sensor_est"]["pos"], processed_data["cartesian_sensor_est"]["vel"]])  # ce: cartesian state of ego
        co = np.vstack([processed_data["obstacle_sensor_est"][obs]["rel_pos"], processed_data["obstacle_sensor_est"][obs]["rel_vel"]]) + ce  # co: cartesian state of the obstacle
        
        n = np.shape(ce)[0]//2

        # It will be better if we have an estimation of the acceleration of the obstacle
        dot_co = np.vstack([co[n:], np.zeros((n,1))])

        phi, p_phi_p_ce, p_phi_p_co = self.phi_and_derivatives(dt, ce, co)

        x =  np.vstack(processed_data["state_sensor_est"]["state"])

        p_ce_p_xe = self._model.jacobian(x)
        
        fx = self._model.fx(x)
        fu = self._model.fu(x)

        d_ce = -self.c * p_phi_p_ce

        u = fu.T @ p_ce_p_xe.T @ d_ce
        
        return phi, u


class ZeroingBarrierFunctionController(EnergyFunctionController):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self._name = 'zeroing_barrier_function'
        self.lambd = spec['lambd']
        
        
    def safe_control(self, u_ref, obs, dt, processed_data):
        """ Compute the safe control between ego and an obstacle.

        Zeroing Barrier Function compute u by solving the following optimization:
        min || u - u_ref ||, 
        s.t.  dot_phi < lambd * phi

        => p_phi_p_xe * dot_xe          + p_phi_p_co * dot_co < lambd * phi
        => p_phi_p_xe * ( fx + fu * u ) + p_phi_p_co * dot_co < lambd * phi
        => p_phi_p_xe * fu * u < lambd * phi - p_phi_p_xe * fx - p_phi_p_co * dot_co

        """
        ce = np.vstack([processed_data["cartesian_sensor_est"]["pos"], processed_data["cartesian_sensor_est"]["vel"]])  # ce: cartesian state of ego
        co = np.vstack([processed_data["obstacle_sensor_est"][obs]["rel_pos"], processed_data["obstacle_sensor_est"][obs]["rel_vel"]]) + ce  # co: cartesian state of the obstacle
        
        n = np.shape(ce)[0]//2

        # It will be better if we have an estimation of the acceleration of the obstacle
        dot_co = np.vstack([co[n:], np.zeros((n,1))])

        phi, p_phi_p_ce, p_phi_p_co = self.phi_and_derivatives(dt, ce, co)

        x =  np.vstack(processed_data["state_sensor_est"]["state"])

        p_ce_p_xe = self._model.jacobian(x)
        
        fx = self._model.fx(x)
        fu = self._model.fu(x)

        p_phi_p_xe = p_ce_p_xe.T @ p_phi_p_ce

        A = cvxopt.matrix(p_phi_p_xe.T @ fu)
        b = cvxopt.matrix(self.lambd * phi - p_phi_p_xe.T @ fx - p_phi_p_co.T @ dot_co)
        A = A / abs(b)
        b = b / abs(b)

        Q = cvxopt.matrix(np.eye(np.shape(u_ref)[0]))
        p = cvxopt.matrix(- 2 * u_ref)
        A = cvxopt.matrix([[A]])
        b = cvxopt.matrix([[b]])

        u = u_ref
        try:
            cvxopt.solvers.options['feastol']=1e-9
            cvxopt.solvers.options['show_progress'] = False
            sol=cvxopt.solvers.qp(Q, p, A, b)
            u = np.vstack(sol['x'])
        except:
            pass

        return phi, u



class SlidingModeController(EnergyFunctionController):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self._name = 'sliding_mode'
        self.c = spec['c']
        
    def safe_control(self, u_ref, obs, dt, processed_data):
        """ Compute the safe control between ego and an obstacle.

        Zeroing Barrier Function compute u by solving the following optimization:

        u = u_ref + c * p_phi_p_xe * fu  when  phi > 0

        c is a large constant

        """
        ce = np.vstack([processed_data["cartesian_sensor_est"]["pos"], processed_data["cartesian_sensor_est"]["vel"]])  # ce: cartesian state of ego
        co = np.vstack([processed_data["obstacle_sensor_est"][obs]["rel_pos"], processed_data["obstacle_sensor_est"][obs]["rel_vel"]]) + ce  # co: cartesian state of the obstacle
        
        n = np.shape(ce)[0]//2

        # It will be better if we have an estimation of the acceleration of the obstacle
        dot_co = np.vstack([co[n:], np.zeros((n,1))])

        phi, p_phi_p_ce, p_phi_p_co = self.phi_and_derivatives(dt, ce, co)

        x =  np.vstack(processed_data["state_sensor_est"]["state"])

        p_ce_p_xe = self._model.jacobian(x)
        
        fx = self._model.fx(x)
        fu = self._model.fu(x)

        p_phi_p_xe = p_ce_p_xe.T @ p_phi_p_ce
    
        
        
        if phi > 0:
            u = u_ref - self.c * fu.T @ p_phi_p_xe
        else:
            u = u_ref
        

        return phi, u


class SublevelSafeSetController(EnergyFunctionController):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self._name = 'sublevel_safe_set'
        self.lambd = spec['lambd']
        
    def safe_control(self, u_ref, obs, dt, processed_data):
        """ Compute the safe control between ego and an obstacle.

        Safe set compute u by solving the following optimization:
        min || u - u_ref ||, 
        s.t.  dot_phi < lambda * dot_phi  or  phi > 0

        => p_phi_p_xe * dot_xe          + p_phi_p_co * dot_co < lambd * phi
        => p_phi_p_xe * ( fx + fu * u ) + p_phi_p_co * dot_co < lambd * phi
        => p_phi_p_xe * fu * u < lambd * phi - p_phi_p_xe * fx - p_phi_p_co * dot_co

        """
        ce = np.vstack([processed_data["cartesian_sensor_est"]["pos"], processed_data["cartesian_sensor_est"]["vel"]])  # ce: cartesian state of ego
        co = np.vstack([processed_data["obstacle_sensor_est"][obs]["rel_pos"], processed_data["obstacle_sensor_est"][obs]["rel_vel"]]) + ce  # co: cartesian state of the obstacle
        
        n = np.shape(ce)[0]//2

        # It will be better if we have an estimation of the acceleration of the obstacle
        dot_co = np.vstack([co[n:], np.zeros((n,1))])

        phi, p_phi_p_ce, p_phi_p_co = self.phi_and_derivatives(dt, ce, co)

        x =  np.vstack(processed_data["state_sensor_est"]["state"])

        p_ce_p_xe = self._model.jacobian(x)
        
        fx = self._model.fx(x)
        fu = self._model.fu(x)

        p_phi_p_xe = p_ce_p_xe.T @ p_phi_p_ce

        L = p_phi_p_xe.T @ fu
        S = self.lambd * phi - p_phi_p_xe.T @ fx - p_phi_p_co.T @ dot_co
        
        u = u_ref

        if phi <= 0 or np.asscalar(L @ u_ref) < np.asscalar(S):
            u = u_ref
        else:
            u = u_ref - (np.asscalar(L @ u_ref - S) * L.T / np.asscalar(L @ L.T))
        
        return phi, u
