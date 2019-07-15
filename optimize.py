import cvxpy as cp
import numpy as np
import pandas as pd


def read_measurements():
	df = pd.read_csv('./measures.csv')
	return df


def solve_optimization():
	df = read_measurements()
	W = 450 * 1000
	U_MAX = 3600 * 24 * 3
	classifiers = list(df['id'])
	weights = list(df['energy_per_sample'])
	values = list(df['val_accuracy'])
	assert len(classifiers) == len(weights) and len(weights) == len(values)
	
	# solving
	selection = cp.Variable(shape=len(weights), integer=True)
	constraints = [weights * selection <= W,
				   selection >= 0.0,
				   cp.sum(selection) == U_MAX]
	objective = cp.Maximize(values * selection)
	problem = cp.Problem(objective=objective, constraints=constraints)
	problem.solve()
	print(problem.status)
	for i, v in enumerate(selection.value):
		print(classifiers[i], int(round(v, 2)))
	return problem.status, selection.value
	


if __name__ == "__main__":
	status, vals = solve_optimization()
	# for v in vals:
	# 	print(int(round(v, 1)))