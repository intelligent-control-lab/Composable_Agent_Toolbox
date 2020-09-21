import argparse

def add_planner_args(parser):
	parser = argparse.ArgumentParser(description='Specification for Planner')
	'''
	planner settings
	'''
	spec = parser.add_argument_group("Planning specification setting")
	spec.add_argument('--dim', default=2, type=int, help='planning dimension')
	spec.add_argument('--horizon', default=9, type=int, help='planning horizon')
	spec.add_argument('--obsr', default=1, type=float, help='obstacle radius')
	spec.add_argument('--n_ob', default=0, type=int, help='the number of obstacles')
	spec.add_argument('--obsp', default=[0, 4], type=float, nargs="+", help='position of obstacle')
	spec.add_argument('--goal', default=[0,9], type=float, nargs="+", help='goal target')
	spec.add_argument('--state', default=[0,0], type=float, nargs="+", help='agent state')
	spec.add_argument('-es', '--experiment-settings', default="settings.yaml", type=str, nargs='+', help='experiment settings yaml file')

	return parser


def add_models_args(parser):
	parser = argparse.ArgumentParser(description='Specification for Planner')
	'''
	planner settings
	'''
	spec = parser.add_argument_group("Planning specification setting")
	spec.add_argument('--dim', default=2, type=int, help='planning dimension')
	spec.add_argument('--horizon', default=9, type=int, help='planning horizon')
	spec.add_argument('--obsr', default=1, type=float, help='obstacle radius')
	spec.add_argument('--n_ob', default=0, type=int, help='the number of obstacles')
	spec.add_argument('--obsp', default=[0, 4], type=float, nargs="+", help='position of obstacle')
	spec.add_argument('--goal', default=[0,9], type=float, nargs="+", help='goal target')
	spec.add_argument('--state', default=[0,0], type=float, nargs="+", help='agent state')
	spec.add_argument('-es', '--experiment-settings', default="settings.yaml", type=str, nargs='+', help='experiment settings yaml file')

	return parser