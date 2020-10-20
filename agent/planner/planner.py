# import math
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import matplotlib
from .src.utils import *
# from src.utils import *
from scipy import interpolate
import math
from math import pi 
from .src.robot import RobotProperty
# from src.robot import RobotProperty


solvers.options['show_progress'] = False

'''Weiye: motion planning'''
class Planner(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.model = model
        self.cache = {}
    def planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        return agent_next_state
    def re_planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        return agent_next_state

    def plan(self, ineq, eq, agent_state):
        pass
    

class OptimizationBasedPlanner(Planner):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self.spec = spec
        self.model = model
        self.replanning_cycle = spec['replanning_cycle']
        self.horizon = spec['horizon']
        self.dim = spec['dim']
        self.cache = {}


    def planning(self, dt, goal, sensor_est_data):
        '''
        planning function to examine the correctness of 3d arm / 2d car planning
        '''
        # 3d arm case 
        if goal['task'] == 'arm_state_reach':
            obs_state = 1
            obs_traj = []
            end_pos = sensor_est_data['cartesian_sensor_est']['pos']
            obs_pos_relative = sensor_est_data['obstacle_sensor_est']['human']
            obs_pos = np.squeeze(end_pos + obs_pos_relative)
            for i in range(self.spec['horizon']):
                tmp = cap()
                tmp.p = np.array([[obs_pos[0], obs_pos[0]], [obs_pos[1], obs_pos[1]], [obs_pos[2], obs_pos[2]]])
                tmp.r = 0.2
                obs_traj.append(tmp)

            # get the current state
            dim = self.dim
            state = sensor_est_data['state_sensor_est']['state'][0:dim,0].reshape([dim,1])

            # only take the planned horizon trajectory
            self.obs_state = obs_state
            self.dt = dt
            self.v_max = 5 # vmax is 5m/s
            
            # hard code goal 
            # goal = np.array([0, 1.5708, 0, 0, 0, 0, 0]).reshape([dim,1])
            # target = goal 
            goal = goal['goal']
            target = goal.reshape([dim,1])         
            # obstacle 
            self.obs_traj = obs_traj
            ref_traj = self.test_plan(self.ineq, self.eq, state, target)
            interp_traj = self.interpolate_joint(ref_traj)

            # calculate the end effector Cartesian position 
            # endeffector_traj = self.endpose_traj(interp_traj)
            # interpolate the compute the velocity 
            # interp_traj_vel = self.pos2vel(endeffector_traj)
            interp_traj_vel = self.pos2vel(interp_traj)
            return interp_traj_vel



        # 2d ball reaching case 
        if goal['task'] == '2d_reach':
            if sensor_est_data["communication_sensor_est"] != {}:
                name = list(sensor_est_data["communication_sensor_est"].keys())[0]
                self.obs_state = sensor_est_data["communication_sensor_est"][name]['state']
            self.dt = dt
            self.v_max = 5 # vmax is 5m/s
            goal = goal['goal']
            target = goal[0:2,0]
            state = sensor_est_data['state_sensor_est']['state'][0:2,0]
            start_vel = np.linalg.norm(sensor_est_data['state_sensor_est']['state'][2:4,0].flatten()) 

            # assignment of time 
            time = self.time_assignment(target, state)
            if sensor_est_data["communication_sensor_est"] != {}:
                obs_traj_est = self.obs_traj_estimate(self.obs_state, time)
            else:
                obs_traj_est = []
            self.obs_traj = obs_traj_est
            ref_traj = self.plan(self.ineq, self.eq, state, target)
            # interp_traj = self.interpolate(ref_traj,start_vel)
            interp_traj = self.interpolate_traj(np.hstack((time,ref_traj)))
            interp_traj_vel = self.pos2vel(interp_traj)
            return interp_traj_vel

    def re_planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        ref_traj = plan(self.ineq, self.eq, agent_next_state, goal)
        return agent_next_state, ref_traj

    def reference_traj_generate(self, agent_state, goal):
        '''
        generate the reference trajectory 
        '''
        # transform to numpy array 
        agent_state = np.array(agent_state)
        goal = np.array(goal)
        traj = np.zeros((self.spec['horizon'], self.spec['dim']))
        traj[[0],:] = agent_state.T
        for i in range(self.spec['horizon']-1):
            traj[[i+1],:] = (agent_state + (i+1) / (self.spec['horizon']-1) * (goal - agent_state)).T
        
        return traj

    def test_plan(self, ineq, eq, agent_state, goal):
        ref_traj = self.reference_traj_generate(agent_state, goal)
        traj = self.CFS_arm(x_ref=ref_traj)
        return traj 

    def plan(self, ineq, eq, agent_state, goal):
        ref_traj = self.reference_traj_generate(agent_state, goal)
        traj = self.CFS(x_ref=ref_traj)
        return traj 

    def ineq(self, x, obs):
        '''
        inequality constraints. 
        constraints: ineq(x) > 0
        '''
        # norm distance restriction
        obs_p = obs.flatten()
        # obs_r = self.spec['obs_r']
        obs_r = 1
        obs_r = np.array(obs_r)
        
        # flatten the input x 
        x = x.flatten()
        dist = np.linalg.norm(x - obs_p) - obs_r
        return dist

    def ineq_arm(self,x,DH,base,obs,cap):
        '''
        inequality constraints. 
        constraints: ineq(x) > 0
        '''
        # obs = np.reshape(obs,(3,1))
        dist = distance_arm(x,DH,base,obs,cap)
        return dist

    def eq(self, x):
        '''
        equality constraints. 
        constraints: eq(x) = 0
        '''
        pass

    def endpose_traj(self, joint_traj):
        '''
        generate the end effector trajctory give the joint trajectory 
        '''
        horizon = joint_traj.shape[0]
        epos_traj = np.empty((horizon,3)) # Cartesian is 3 dimensional 
        for i in range(horizon):
            epos = forkine(joint_traj[i,:].transpose(),self.robot.DH,self.robot.base)
            epos_traj[i,:] = epos.transpose()
        return epos_traj



    def time_assignment(self, goal, start):
        v_max = self.v_max
        dist = np.linalg.norm((goal - start))
        total_time = dist / v_max
        time_interval = total_time / (self.spec['horizon'] - 1)
        time = np.zeros((self.spec['horizon'], 1))
        tmp = 0
        for i in range(self.spec['horizon']):
            time[i,:] = tmp
            tmp += time_interval
        return time


    def obs_traj_estimate(self, obs_state, time):
        '''
        estimate the future obstacle position
        '''
        obs_traj = np.zeros((self.spec['horizon'], self.spec['dim']))
        pos = obs_state[0:2,:]
        vel = obs_state[2:4,:]
        for i in range(self.spec['horizon']):
            pos_tmp = pos + vel*time[i,:]
            obs_traj[i,:] = pos_tmp.transpose()
        return obs_traj


    def interpolate_joint(self, x_ref):
        '''
        to interpolate the planned trajectory in joint space
        '''
        dt = self.dt
        v_max = self.robot.vmax # 7 joint 
        time_max_list = [0]
        # check the longest joint travel time between two points 
        for i in range(self.horizon-1):
            pos1 = x_ref[i,:]
            pos2 = x_ref[i+1,:]
            dist = np.abs(pos2 - pos1) 
            time = dist / v_max
            time_max = np.amax(time)
            time_max_list.append(time_max + time_max_list[-1]) 

        # compute t_list length 
        tn = int((time_max_list[-1]) / dt) # num of stamps
        t_list = np.linspace(0, tn*dt, tn+1)

        # generate interpolated trajectory
        path_interp = np.empty((len(t_list), x_ref.shape[1]))
        for i in range(x_ref.shape[1]):
            f_interp = interpolate.interp1d(time_max_list, x_ref[:, i], kind = 'slinear') # "nearest","zero","slinear","quadratic","cubic"
            path_interp[:,i] = f_interp(t_list)
        path_interp = np.vstack((path_interp, x_ref[-1,:])) # append the last point
        return path_interp


    def interpolate(self, x_ref, v0):
        '''
        to interpolate the planned trajectory 
        '''
        # determine how many points to interpolate 
        # determine the velocity 

        # parameters
        # v_max = 5 # 2m/s
        v_max = self.v_max
        a_max = 10 # 1m/s^2
        dt = self.dt

        # interpolate 
        # step 1: calculate the displacement
        s_origin = np.squeeze(np.zeros((self.horizon,1)))
        # s_origin = np.zeros((1,3))
        for i in range(self.horizon-1):
            pos1 = x_ref[i,:]
            pos2 = x_ref[i+1,:]
            dist = np.linalg.norm(pos2 - pos1)
            s_origin[i+1] = dist + s_origin[i]

        s_max = s_origin[-1]

        # step 2: switch state
        t1 = abs(v0 - v_max) / a_max
        s1 = 0.5 * (v0 + v_max) * t1
        t3 = v_max / a_max
        s3 = 0.5 * v_max * t3
        if s1 + s3 < s_max: # have constant speed segment
            have_const_spd = True
        else: # no constant speed segment
            have_const_spd = False
            if v0 >= v_max: # deccelerate till the end
                always_dec = True
            else: # accelerate to v_top and deccelerate
                always_dec = False

        # step 3: calculate the time sequence
        if have_const_spd: # have constant speed segment
            s2 = s_max - s1 - s3
            t2 = s2 / v_max
            tn = int((t1 + t2 + t3) / dt) # num of stamps
            t_list = np.linspace(0, tn*dt, tn+1)
        else: # no constant speed segment
            if always_dec: # deccelerate till the end
                v_top = v_max
                t1 = (v0 - v_top) / a_max
                s1 = 0.5 * (v0 + v_top) * t1
                t3 = v_top / a_max
                s3 = 0.5 * v_top * t3
            else: # accelerate to v_top and deccelerate
                v_top = math.sqrt(2 * a_max * s_max + 0.5 * v0 ** 2)
                t1 = (v_top - v0) / a_max
                s1 = 0.5 * (v0 + v_top) * t1
                t3 = v_top / a_max
                s3 = 0.5 * v_top * t3
            
            tn = int((t1 + t3) / dt) # num of stamps
            t_list = np.linspace(0, tn*dt, tn+1)


        # step 4: compute interpolated displacement sequence
        s_interp = np.zeros(t_list.shape)
        if have_const_spd: # have constant speed segment
            for i in range(len(s_interp)):
                if t_list[i] < t1:
                    t_tmp = t_list[i]
                    s_interp[i] = v0 * t_tmp + np.sign(v_max - v0) * 0.5 * a_max * t_tmp**2
                elif t_list[i] >= t1 and t_list[i] <= t1 + t2:
                    t_tmp = t_list[i] - t1
                    s_interp[i] = s1 + v_max * t_tmp
                else:
                    t_tmp = t_list[i] - t1 - t2
                    s_interp[i] = s1 + s2 + v_max * t_tmp - 0.5 * a_max * t_tmp**2
        else: # no constant speed segment
            for i in range(len(s_interp)):
                if t_list[i] < t1:
                    t_tmp = t_list[i]
                    s_interp[i] = v0 * t_tmp + np.sign(v_max - v0) * 0.5 * a_max * t_tmp**2
                else:
                    t_tmp = t_list[i] - t1
                    s_interp[i] = s1 + v_top * t_tmp - 0.5 * a_max * t_tmp**2

        # step 5: generate interpolated trajectory
        path_interp = np.empty((len(t_list), x_ref.shape[1]))
        for i in range(x_ref.shape[1]):
            f_interp = interpolate.interp1d(s_origin, x_ref[:, i], kind = 'slinear') # "nearest","zero","slinear","quadratic","cubic"
            path_interp[:,i] = f_interp(s_interp)
        path_interp = np.vstack((path_interp, x_ref[-1,:]))
        return path_interp


    def interpolate_traj(self, x_ref):
        '''
        to interpolate the planned trajectory using simple linear interpolation
        input: x_ref:
                col1: t;  col2: x;  col3: y
        output: traj_interp:
                col1: x_interp;  col2: y_interp
        '''
        # determine how many points to interpolate 
        # determine the velocity 

        # parameters
        dt = self.dt

        # interpolate 
        # step 1: calculate the interpolated time list
        t_end = x_ref[-1,0]
        tn = int(t_end / dt)
        t_list = np.linspace(0, tn*dt, tn+1)

        # step 2: generate interpolated trajectory
        traj_interp = np.empty((len(t_list), x_ref.shape[1]-1))
        for i in range(x_ref.shape[1]-1):
            f_interp = interpolate.interp1d(x_ref[:, 0], x_ref[:, i+1], kind = 'slinear') # "nearest","zero","slinear","quadratic","cubic"
            traj_interp[:,i] = f_interp(t_list)
        traj_interp = np.vstack((traj_interp, x_ref[-1,1:3]))
        return traj_interp


    def CFS(
        self, 
        x_ref,
        cq = [10,0,10], 
        cs = [0,1,0.1], 
        minimal_dis = 0, 
        ts = 1, 
        maxIter = 30,
        stop_eps = 1e-3
    ):
        # without obstacle, then collision free
        
        n_ob = self.spec['n_ob']
        obs_traj = self.obs_traj
        if n_ob == 0 or len(obs_traj)==0: # no future obstacle information is provided 
            return np.array(x_ref)

        # has obstacle, the normal CFS procedure 
        x_rs = np.array(x_ref)

        # planning parameters 
        h = x_rs.shape[0]    
        dimension = x_rs.shape[1] #

        # flatten the trajectory to one dimension
        # flatten to one dimension for applying qp, in the form of x0,y0,x1,y1,...
        x_rs = np.reshape(x_rs, (x_rs.size, 1))
        x_origin = x_rs

        # objective terms 
        # identity
        Q1 = np.identity(h * dimension)
        S1 = Q1
        # velocity term 
        Vdiff = np.identity(h*dimension) - np.diag(np.ones((1,(h-1)*dimension))[0],dimension)
        Q2 = np.matmul(Vdiff.transpose(),Vdiff) 
        # Acceleration term 
        Adiff = Vdiff - np.diag(np.ones((1,(h-1)*dimension))[0],dimension) + np.diag(np.ones((1,(h-2)*dimension))[0],dimension*2)
        Q3 = np.matmul(Adiff.transpose(),Adiff)
        # Vdiff = eye(nstep*dim)-diag(ones(1,(nstep-1)*dim),dim);

        # objective 
        Q = Q1*cq[0]+Q2*cq[1]+Q3*cq[2];
        S = S1*cs[0]+Q2*cs[1]+Q3*cs[2];

        # quadratic term
        H =  Q + S 
        # linear term
        f = -1 * np.dot(Q, x_origin)

        b = np.ones((h * n_ob, 1)) * (-minimal_dis)
        H = matrix(H,(len(H),len(H[0])),'d')
        f = matrix(f,(len(f), 1),'d')
        # b = matrix(b,(len(b),1),'d')

        # reference trajctory cost 
        J0 =  np.dot(np.transpose(x_rs - x_origin), np.dot(Q, (x_rs - x_origin))) + np.dot(np.transpose(x_rs), np.dot(S, x_rs))
        J = float('inf')
        dlt = float('inf')
        cnt = 0

        # equality constraints 
        # start pos and end pos remain unchanged 
        Aeq = np.zeros((dimension*2, len(x_rs)))
        for i in range(dimension):
            Aeq[i,i] = 1
            Aeq[dimension*2-i-1, len(x_rs)-i-1] = 1
        
        beq = np.zeros((dimension*2, 1))
        beq[0:dimension,0] = x_rs[0:dimension,0]
        beq[dimension:dimension*2, 0] = x_rs[dimension*(h-1): dimension*h, 0] 
        # transform to convex optimization matrix 
        Aeq = matrix(Aeq,(len(Aeq),len(Aeq[0])),'d')
        beq = matrix(beq,(len(beq),1),'d')

        # set the safety margin 
        D = 5

        # fig, ax = plt.subplots()
        # main CFS loop
        while dlt > stop_eps:
            cnt += 1
            Lstack, Sstack = [], []
            # inequality constraints 
            # l * x <= s
            Constraint = np.zeros((h * n_ob, len(x_rs)))
            
            for i in range(h):
                # get reference pos at time step i
                if i < h-1 and i > 0:
                    x_r = x_rs[i * dimension : (i + 1) * dimension] 

                    # get inequality value (distance)
                    # get obstacle at this time step 
                    obs_p = obs_traj[i,:]  
                    dist = self.ineq(x_r,obs_p)
                    # print(dist)

                    # get gradient 
                    ref_grad = jac_num(self.ineq, x_r, obs_p)
                    # print(ref_grad)

                    # compute
                    s = dist - D - np.dot(ref_grad, x_r)
                    l = -1 * ref_grad
                if i == h-1 or i == 0: # don't need inequality constraints for lst dimension 
                    s = np.zeros((1,1))
                    l = np.zeros((1,2))

                # update 
                Sstack = vstack_wrapper(Sstack, s)
                l_tmp = np.zeros((1, len(x_rs)))
                l_tmp[:,i*dimension:(i+1)*dimension] = l
                Lstack = vstack_wrapper(Lstack, l_tmp)

            Lstack = matrix(Lstack,(len(Lstack),len(Lstack[0])),'d')
            Sstack = matrix(Sstack,(len(Sstack),1),'d')

            # QP solver 
            sol = solvers.qp(H, f, Lstack, Sstack, Aeq, beq)
            x_ts = sol['x']
            x_ts = np.reshape(x_ts, (len(x_rs),1))

            J = np.dot(np.transpose(x_ts - x_origin), np.dot(Q, (x_ts - x_origin))) + np.dot(np.transpose(x_ts), np.dot(S, x_ts))
            dlt = min(abs(J - J0), np.linalg.norm(x_ts - x_rs))
            J0 = J
            x_rs = x_ts
            if cnt >= maxIter:
                break
        
        # return the reference trajectory      
        x_rs = x_rs[: h * dimension]
        return x_rs.reshape(h, dimension)

    def CFS_arm(
        self, 
        x_ref,
        cq = [1,10,0], 
        cs = [0,0,20], 
        minimal_dis = 0, 
        ts = 1, 
        maxIter = 6,
        stop_eps = 1e-1
    ):
        
        # define the robot here 
        robot = RobotProperty()
        self.robot = RobotProperty()

        # without obstacle, then collision free
        n_ob = self.spec['n_ob']
        obs_traj = self.obs_traj
        if n_ob == 0 or len(obs_traj)==0: # no future obstacle information is provided 
            print(f"{len(obs_traj)} length obstacle, direct exit!!!!")
            return x_ref

        # has obstacle, the normal CFS procedure 
        x_rs = np.array(x_ref)

        # planning parameters 
        h = x_rs.shape[0]    
        dimension = x_rs.shape[1] #

        # flatten the trajectory to one dimension
        # flatten to one dimension for applying qp, in the form of x0,y0,x1,y1,...
        x_rs = np.reshape(x_rs, (x_rs.size, 1))
        x_origin = x_rs

        # objective terms 
        # identity
        Q1 = np.identity(h * dimension)
        # distance metric 
        qd = np.array([[1, 0, 0, 0, 0, 0, 0],
                        [0, 0.01, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 4, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 4]])
        for i in range(h):
            Q1[i*dimension:(i+1)*dimension,i*dimension:(i+1)*dimension] = qd*0.1
            if i == h-1:
                Q1[i*dimension:(i+1)*dimension,i*dimension:(i+1)*dimension] = qd
        S1 = Q1
        # velocity term 
        Vdiff = np.identity(h*dimension) - np.diag(np.ones((1,(h-1)*dimension))[0],dimension)
        # Q2 = np.matmul(Vdiff.transpose(),Vdiff) 
        Q2 = np.matmul(Vdiff[0:(h-1)*dimension,:].transpose(),Vdiff[0:(h-1)*dimension,:])
        # Acceleration term 
        Adiff = Vdiff - np.diag(np.ones((1,(h-1)*dimension))[0],dimension) + np.diag(np.ones((1,(h-2)*dimension))[0],dimension*2)
        Q3 = np.matmul(Adiff[0:(h-2)*dimension,:].transpose(),Adiff[0:(h-2)*dimension,:])
        # Vdiff = eye(nstep*dim)-diag(ones(1,(nstep-1)*dim),dim);

        # objective 
        Q = Q1*cq[0]+Q2*cq[1]+Q3*cq[2];
        S = S1*cs[0]+Q2*cs[1]+Q3*cs[2];

        # quadratic term
        H =  Q + S 
        # linear term
        f = -1 * np.dot(Q, x_origin)
        b = np.ones((h * n_ob, 1)) * (-minimal_dis)
        H = matrix(H,(len(H),len(H[0])),'d')
        f = matrix(f,(len(f), 1),'d')
        # b = matrix(b,(len(b),1),'d')

        # reference trajctory cost 
        J0 =  np.dot(np.transpose(x_rs - x_origin), np.dot(Q, (x_rs - x_origin))) + np.dot(np.transpose(x_rs), np.dot(S, x_rs))
        J = float('inf')
        dlt = float('inf')
        cnt = 0

        # equality constraints 
        # start pos and end pos remain unchanged 
        Aeq = np.zeros((dimension*2, len(x_rs)))
        for i in range(dimension):
            Aeq[i,i] = 1
            Aeq[dimension*2-i-1, len(x_rs)-i-1] = 1
        
        beq = np.zeros((dimension*2, 1))
        beq[0:dimension,0] = x_rs[0:dimension,0]
        beq[dimension:dimension*2, 0] = x_rs[dimension*(h-1): dimension*h, 0] 
        # transform to convex optimization matrix 
        Aeq = matrix(Aeq,(len(Aeq),len(Aeq[0])),'d')
        beq = matrix(beq,(len(beq),1),'d')


        # main CFS loop
        while dlt > stop_eps:
            cnt += 1
            Lstack, Sstack = [], []
            # inequality constraints 
            # l * x <= s
            Constraint = np.zeros((h * n_ob, len(x_rs)))
            for i in range(h):
                # get reference pos at time step i
                if i < h-1 and i > 0:
                    x_r = x_rs[i * dimension : (i + 1) * dimension] 

                    # get inequality value (distance)
                    # get obstacle at this time step  
                    obs = obs_traj[i]  
                    dist = self.ineq_arm(x_r,robot.DH,robot.base,obs,robot.cap)

                    # get gradient 
                    ref_grad = jac_num_arm(self.ineq_arm, x_r,robot.DH,robot.base,obs,robot.cap)

                    # compute
                    s = dist - robot.margin - np.dot(ref_grad, x_r)
                    l = -1 * ref_grad
                if i == h-1 or i == 0: # don't need inequality constraints for lst dimension 
                    s = np.zeros((1,1))
                    l = np.zeros((1,self.dim))

                # update 
                Sstack = vstack_wrapper(Sstack, s)
                l_tmp = np.zeros((1, len(x_rs)))
                l_tmp[:,i*dimension:(i+1)*dimension] = l
                Lstack = vstack_wrapper(Lstack, l_tmp)

            # add joint limit 
            # l_lim_u = np.identity(h*dimension)
            # l_lim_l = -1*np.identity(h*dimension)
            # Lstack = vstack_wrapper(Lstack, l_lim_u)
            # Lstack = vstack_wrapper(Lstack, l_lim_l)
            # Sstack = vstack_wrapper(Sstack, 2*pi*np.ones((h*dimension,1)))
            # # Sstack = vstack_wrapper(Sstack, -pi*np.ones((60,1)))
            # Sstack = vstack_wrapper(Sstack, np.zeros((h*dimension,1)))
            Lstack = matrix(Lstack,(len(Lstack),len(Lstack[0])),'d')
            Sstack = matrix(Sstack,(len(Sstack),1),'d')

            # QP solver 
            sol = solvers.qp(H, f, Lstack, Sstack, Aeq, beq)
            x_ts = sol['x']
            x_ts = np.reshape(x_ts, (len(x_rs),1))

            J = np.dot(np.transpose(x_ts - x_origin), np.dot(Q, (x_ts - x_origin))) + np.dot(np.transpose(x_ts), np.dot(S, x_ts))
            dlt = min(abs(J - J0), np.linalg.norm(x_ts - x_rs))
            J0 = J
            x_rs = x_ts
            if cnt >= maxIter:
                break
            
        
        # return the reference trajectory     
        x_rs = x_rs[: h * dimension]
        return x_rs.reshape(h, dimension)

    def pos2vel(self, traj):
        '''
        get the velocity reference from the position reference
        '''
        ref_traj = []
        horizon = traj.shape[0]
        for i in range(horizon-1):
            pos1 = traj[i,:]
            pos2 = traj[i+1,:]
            diff = pos2 - pos1
            vel = diff / self.dt
            traj_tmp = np.hstack((pos1, vel))
            ref_traj = vstack_wrapper(ref_traj, traj_tmp)
        # concatenate the final pos
        dim = traj.shape[1]
        ref_traj = vstack_wrapper(ref_traj, np.hstack((traj[horizon-1,:],np.zeros(dim))))
        return ref_traj


class SamplingBasedPlanner(Planner):
    def __init__(self, spec, model):
        super().__init__(self, spec, model)
        self.spec = spec
        self.model = model
        self.cache = {}

    def planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        return agent_next_state

    def re_planning(self, dt, goal, agent_state):
        agent_next_state = agent_state
        return agent_next_state

    def plan(self, ineq, eq, agent_state):
        pass



if __name__ == '__main__':
    # from .src.configs import add_planner_args
    from src.configs import add_planner_args
    from pprint import pprint
    import argparse
    
    '''
    test the planner class
    test the convex feasible set algorithm
    '''
    parser = argparse.ArgumentParser(description='Planning Parameters Setting')
    parser = add_planner_args(parser)
    args, unknown = parser.parse_known_args()
    args = vars(args)

    experiment_settings = load_experiment_settings(args['experiment_settings'])
    args.update(experiment_settings)

    pprint(args)

    '''
    test case for 3d arm motion planning 
    '''
    obs_state = 1
    obs_traj = []
    for i in range(args['horizon']):
        tmp = cap()
        tmp.p = np.array([[0.2, 0.2], [0.2, 0.2], [0.7, 0.7]])
        tmp.p = np.array([[0.2, 0.2], [0.2, 0.2], [0.7, 0.7]])
        tmp.r = 0.2
        obs_traj.append(tmp)
        # obs_traj[i,:] = np.array([0.45 -0.95 1.21]) # the obstacle position

    # models 
    model = 1 # the place holder 
    CFS  = OptimizationBasedPlanner(args, model)
    dt = 0.02

    # read the start and goal position 
    start = np.zeros((7,1))
    goal = np.zeros((7,1))
    for i in range(7):
        start[i] = float(args['start'][i])
        goal[i] = float(args['goal'][i])

    ref_traj = CFS.test_planning(dt, goal, start, obs_traj, obs_state)
    print(ref_traj) 

    




