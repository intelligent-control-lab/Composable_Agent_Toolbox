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
sys.path.append("../../..") # Adds higher directories to python modules path
import agent.model.models.manipulator_dyn as manipulator_dyn  #pylint will throw a fit here because of line 15

# Start the actual test

spec2 = {
	   "use_spec"      : 0,
    "use_library"   : 0,
    "model_name"    : '2link_arm',
    "time_sample"   : 0.01,
    "disc_flag"     : 1,
    "model_spec"    : 0
} # Define specifications that initialize the model

# Start the actual test!
print("Test is Starting! Hold on!!!!")
pen_model2 = manipulator_dyn.ManipulatorEquation(spec2)