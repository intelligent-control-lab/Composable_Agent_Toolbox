# AUTH: NOREN
# DESC: This is a test file for use/understanding of how to work with the model class
# SUMM: Implement a nonlinear pendulum model in a class and then does some evaluations with it

# Python Standard Lib imports
import math as m
import time
import sys

# Third Party Imports
import numpy as np
import sympy as sp

# Project-specific Imports
sys.path.append("../models/") # Adds higher directories to python modules path
import nonlinmodel_dyn #pylint will throw a fit here because of line 15

# Start the actual test

# Declare a system model spec, NOTE THAT ORDER DOES MATTER (SYMS |--> FUNCS |--> SPEC)
# Note: mss stands for model_spec_syms
mss = {
	   "states"       : sp.symbols(['x1', 'x2']),
	   "state_dict"   : {'x1':'angle', 'x2':'angular velocity'},
	   "cntls"	      : sp.symbols(['u1']),
	   "cntl_dict"    : {'u1':'torque'},
	   "params"       : sp.symbols(['m', 'l', 'g']),
	   "param_dict"   : {'m':'mass', 'l': 'string length', 'g': 'acceleration due to gravity'},
} # Specify the model parameters, which are loaded first

# Note: msf stands for model_spec_functions
msf = {
	   "f_expr"	      : sp.Matrix([[mss["states"][1]], [-1*sp.sin(mss["states"][0])*mss["params"][2]/mss["params"][1]]]),
	   "g_expr"		  : sp.Matrix([[0], [1/(mss["params"][0]*mss["params"][1]**2)]]),
	   "m_func"		  : sp.eye(sp.Matrix(mss["states"]).shape[0])*sp.Matrix(mss["states"])
} # Specify the model functions, which are loaded second

# Combine earlier dicts to align with model lib standards.
model_spec= {
	   "syms"  : mss,
	   "funcs" : msf
} # Specify the model specs that you want to load directly - This is what is actually passed to the model

# Initialization Specification
spec = {
       "use_spec"      : 1,
       "use_library"   : 0,
       "model_name"    : 'nonlin_pen',
       "time_sample"   : 0.01,
       "disc_flag"     : 1,
       "model_spec"    : model_spec
} # Define specifications that initialize the model

spec2 = {
	"use_spec"      : 0,
       "use_library"   : 0,
       "model_name"    : 'nonlin_pen',
       "time_sample"   : 0.01,
       "disc_flag"     : 1,
       "model_spec"    : model_spec
} # Define specifications that initialize the model

spec3 = {
	"use_spec"      : 1,
       "use_library"   : 1,
       "model_name"    : 'nonlin_pen',
       "time_sample"   : 0.01,
       "disc_flag"     : 1,
       "model_spec"    : 'test_file_nonlinpen'
} # Define specifications that initialize the model

# Start the actual test!
print("Test is Starting! Hold on!!!!")
pen_model = nonlinmodel_dyn.NonlinModelCntlAffine(spec)
pen_model2 = nonlinmodel_dyn.NonlinModelCntlAffine(spec2)
pen_model3 = nonlinmodel_dyn.NonlinModelCntlAffine(spec3)
print(pen_model.model_name) # print out the model name
print(pen_model.u) # print out the symbolic vector of model control
print(pen_model.cont_model) # print out the continous dynamic equations
print(pen_model.disc_model) # print out the discrete time dynamic equations
print(pen_model.measure_func) # print out the measurement function

print('Before evaluating the model, let us test the linearization scheme')
print(pen_model.lin_model)
pen_model.linearize_dynamics(x_sub=[m.pi, 1], u_sub=[0.01])
print('This is the after')
print(pen_model.lin_model)
print(' ')

print('Testing the process time')
start1 = time.process_time() # get an initial start time to check to the see the computational time needed for subs-based system evaluation
evaled = pen_model.evaluate_dynamics([m.pi, 1], [0.01], [2, 9.81, 9.81]) # Evaluate using the subs command
end1 = (time.process_time()-start1) # get the end time to check computational cost
print(' ') # Blank line
print('Number of Seconds for Subs Method')
print(end1)    
print('Value of the state when placed in the dynamics')
print(evaled)
print('Value of the expected measurement equation')
evaled_measure = pen_model.evaluate_measurement([m.pi, 1])
print(evaled_measure)
print(pen_model.disc_model_lam) # print out the object of the lambda function - should give the memory address and the type of variable (lambdified-generated mathematical expression)
start2 = time.process_time()
evaled2 = pen_model.disc_model_lam([[m.pi, 1], [0.01], [2, 9.81, 9.81]]) # Evaluate using the lambdify-generated functions
end2 = time.process_time() - start2
print(evaled2)
print(type(evaled2))
print('Number of Seconds for lambda function method')
print(end2)
print('Lambda Function')
print(pen_model.measure_func_lam)
ans = pen_model.measure_func_lam([[m.pi, 1], [0.01], [2, 9.81, 9.81]])
print(ans) # Example of the evaluation of the lambdify-generated measurement equations
print(ans.shape)
print('The type of Ans')
print(type(ans))
print('Cast')
ans2 = np.array(ans)
print('Print Out')
ans2
print(isinstance(ans2, np.ndarray))
print('List of Symbols')
print(pen_model.symbol_dict)

evaled22 = pen_model2.disc_model_lam([[m.pi, 1], [0.01], [2, 9.81, 9.81]]) # Evaluate using the lambdify-generated functions
evaled33 = pen_model3.disc_model_lam([[m.pi, 1], [0.01], [2, 9.81, 9.81]]) # Evaluate using the lambdify-generated functions
print("Difference between Pendulum Model 1 and 2")
delta = evaled22 - evaled2
print(delta)
print("Difference between Pendulum Model 1 and 3")
delta = evaled33 - evaled2
print(delta)
print("Difference between Pendulum Model 2 and 3")
delta = evaled33 - evaled22
print(delta)