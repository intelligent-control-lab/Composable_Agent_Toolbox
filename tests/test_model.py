import math as m
import time 
import sys
import model

# Start the actual test
def test_model_init():
    # Declare a system model:
    spec = {
        "use_library"   : 0,
        "model_name"    : 'nonlin_pen',
        "time_sample"   : 0.01,
        "disc_flag"     : 1
    }
    pen_model = model.NonlinModelCntlAffine(spec)
    assert pen_model.use_library == spec["use_library"]
    assert pen_model.model_name == spec["model_name"]
    assert pen_model.time_sample == spec["time_sample"]
    assert pen_model.disc_flag == spec["disc_flag"]
