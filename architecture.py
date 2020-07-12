from utils.dependency_util import show_architecture

if __name__ == "__main__":
    specified = {
        "controller":{"PID":{"x":"cartesian_x", "goal_x":"cartesian_goal_x"}},
        "planner":{"OptimizationBasedPlanner":{"x":"cartesian_x", "goal_x":"cartesian_goal_x"}}, 
        "model":{"ModelBase":{}},
        "estimator":{"NaiveEstimator":{}},
        "sensor":{"StateSensor":{}, "CartesianSensor":{}}
    }

    show_architecture(specified)
