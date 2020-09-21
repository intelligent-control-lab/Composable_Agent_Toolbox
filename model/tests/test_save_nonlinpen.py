# Python Standard Lib imports
import math as m
import time
import sys

# Third Party Imports
import numpy as np
import sympy as sp
import dill

# Project-specific Imports
sys.path.append("../models") # Adds higher directories to python modules path
import nonlinmodel_dyn #pylint will throw a fit here because of line 15


# Start the actual test

# Declare a system model spec, NOTE THAT ORDER DOES MATTER (SYMS |--> FUNCS |--> SPEC)
# Note: mss stands for model_spec_syms
mss = {
       "states"       : sp.symbols(['x1', 'x2']),
       "state_dict"   : {'x1':'angle', 'x2':'angular velocity'},
       "cntls"        : sp.symbols(['u1']),
       "cntl_dict"    : {'u1':'torque'},
       "params"       : sp.symbols(['m', 'l', 'g']),
       "param_dict"   : {'m':'mass', 'l': 'string length', 'g': 'acceleration due to gravity'},
} # Specify the model parameters, which are loaded first

# Note: msf stands for model_spec_functions
msf = {
       "f_expr"       : sp.Matrix([[mss["states"][1]], [-1*sp.sin(mss["states"][0])*mss["params"][2]/mss["params"][1]]]),
       "g_expr"       : sp.Matrix([[0], [1/(mss["params"][0]*mss["params"][1]**2)]]),
       "m_func"       : sp.eye(sp.Matrix(mss["states"]).shape[0])*sp.Matrix(mss["states"])
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

# Declare a Pendulum Model
pen_model = nonlinmodel_dyn.NonlinModelCntlAffine(spec)

print(pen_model.symbol_dict)
lib_pen_mod = {"states": pen_model.x, "state_dict": pen_model.state_dict, "cntls": pen_model.u, "cntl_dict": pen_model.cntl_dict, "params": pen_model.params, "param_dict": pen_model.param_dict, "Model_CT": pen_model.cont_model, "Measure Function": pen_model.measure_func}
# Make a Toy Model just so that we have multiple models in the library:
a = sp.Symbol('a')
b = sp.Symbol('b')

y = a + 3*b
test_dict = {"state": "test_1", "control": "test_2", "params": "test_3"}

# Declare a Library Object
stored_lib = {"System A": {"model": y, "dictionaries": test_dict}}

# Add our model to the library
stored_lib["nonlin_pen"] = lib_pen_mod


# This actually does the Dill dump.
test_file = open('test_file_nonlinpen', 'wb')
print('Storing the Data')
dill.dump(stored_lib, test_file)
test_file.close()