# AUTH: NOREN
# DESC: This is a test file for use/understanding of how to work with the model class
# SUMM: This file implements a nonlinear pendulum model in a class and then does some evaluations with it

# Python Standard Lib imports
import math as m
import time 
import sys

# Third Party Imports
# None

sys.path.append("..") # Adds higher directories to python modules path
import model

# Start the actual test

# Declare additional parameters for system model:
spec = {
    "use_library"   : 0,
    "model_name"    : 'Ballbot',
    "time_sample"   : 0.01,
    "disc_flag"     : 1
}

ball_model = model.LinearModel(spec)
print(ball_model.model_name) # print out the model name
print(ball_model.u) # print out the symbolic vector of model control
print(ball_model.cont_model) # print out the continous dynamic equations
print(ball_model.disc_model) # print out the discrete time dynamic equations
print(ball_model.measure_func) # print out the measurement function

print('Testing the process time')
start1 = time.process_time() # get an initial start time to check to the see the computational time needed for subs-based system evaluation
evaled = ball_model.evaluate_dynamics([1, 2, 3, 4], [1,1], [2]) # Evaluate using the subs command
end1 = (time.process_time()-start1) # get the end time to check computational cost
print(' ') # Blank line
print('Number of Seconds for Subs Method')
print(end1)    
print('Value of the state when placed in the dynamics')
print(evaled)
print('Value of the expected measurement equation')
evaled_measure = ball_model.evaluate_measurement([1, 2, 3, 4])
print(evaled_measure)
print(ball_model.disc_model_lam) # print out the object of the lambda function - should give the memory address and the type of variable (lambdified-generated mathematical expression)
start2 = time.process_time()
evaled2 = ball_model.disc_model_lam([[1, 2, 3, 4], [1,1], [2]]) # Evaluate using the lambdify-generated functions
end2 = time.process_time() - start2
print(evaled2)
print(type(evaled2))
print('Number of Seconds for lambda function method')
print(end2)
print('Lambda Function')
print(ball_model.measure_func_lam)
print(ball_model.measure_func_lam([[1, 2, 3, 4], [1,1], [2]])) # Example of the evaluation of the lambdify-generated measurement equations
print('List of Symbols')
print(ball_model.symbol_dict)