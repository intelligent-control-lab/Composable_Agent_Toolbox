# AUTH: NOREN
# DESC: This is a test file for use/understanding of how to work with the model class
# SUMM: This file implements a nonlinear pendulum model in a class and then does some evaluations with it

# Python Standard Lib imports
import math as m
import time 
import sys

# Third Party Imports
# None

# Project-specific Imports

import model

# Start the actual test

# Declare a system model:
spec = {
    "use_library"   : 0,
    "model_name"    : 'nonlin_pen',
    "time_sample"   : 0.01,
    "disc_flag"     : 1
}
pen_model = model.NonlinModelCntlAffine(spec)
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
print(pen_model.measure_func_lam([[m.pi, 1], [0.01], [2, 9.81, 9.81]])) # Example of the evaluation of the lambdify-generated measurement equations
print('List of Symbols')
print(pen_model.symbol_dict)