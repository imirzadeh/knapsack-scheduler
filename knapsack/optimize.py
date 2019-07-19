import cvxpy as cp
import numpy as np
import pandas as pd
from knapsack import helpers
from knapsack.settings import RESULT_FILE


def read_measurements_file():
	df = helpers.read_sheet(RESULT_FILE, 'measures')
	return df

	
def prepare_optimization_problem_variables():
	df = read_measurements_file()
	ids = list(df['id'])
	weights = list(df['energy_per_sample'])
	values = list(df['score'])
	dataset_name = get_dataset_name(df)
	
	if helpers.is_regression_dataset(dataset_name):
		values = helpers.one_minus_x(helpers.scale_values(values))
	
	assert len(ids) == len(weights) and len(weights) == len(values)
	
	return {
		'weights': weights,
		'values': values,
		'ids': ids,
	}


def get_dataset_name(df):
	return list(df['dataset'])[0]


def solve_knapsack(W, U_MAX):
	# get variables from file
	variables = prepare_optimization_problem_variables()
	weights = variables['weights']
	values = variables['values']
	ids = variables['ids']
	
	# solving
	selection = cp.Variable(shape=len(weights), integer=True)
	constraints = [weights * selection <= W,
				   selection >= 0.0,
				   cp.sum(selection) == U_MAX]
	objective = cp.Maximize(values * selection)
	problem = cp.Problem(objective=objective, constraints=constraints)
	problem.solve()

	return problem.status, ids, selection.value
	

def report_solution(U_MAX):
	df = read_measurements_file()
	is_regression = helpers.is_regression_dataset(get_dataset_name(df))
	
	min_energy = U_MAX * min(list(df['energy_per_sample'])) + 1100
	max_energy = U_MAX * max(list(df['energy_per_sample']))
	NUM_W = 20
	delta = (max_energy - min_energy) // NUM_W
	W_list = [((min_energy + i * delta)//1000)*1000 for i in range(NUM_W)]
	# W_list = [50000, 75000, 100000, 150000, 200000]
	
	optimal_values = pd.DataFrame()
	sheets = {
		'measure': df.set_index('id'),
		
	}
	
	best_accuracy_index = df['score'].idxmin() if is_regression else df['score'].idxmax()
	min_energy_index = df['energy_per_sample'].idxmin()
	best_accuracy_model = df.loc[best_accuracy_index]
	min_energy_model = df.loc[min_energy_index]
	
	comparison_data = [
		{'model': 'best_score-' + str(best_accuracy_model['id']), 'score': best_accuracy_model['score'], 'energy': (U_MAX*best_accuracy_model['energy_per_sample'])/1000},
		{'model': 'min_energy-' + str(min_energy_model['id']), 'score': min_energy_model['score'], 'energy': (U_MAX*min_energy_model['energy_per_sample'])/1000}
	]
	
	for W in W_list:
		status, ids, solution = solve_knapsack(W, U_MAX)
		if status == cp.INFEASIBLE or status == cp.INFEASIBLE_INACCURATE:
			comparison_data.append({
				'model': '{}-optimal'.format(W//1000),
				'score': None,
				'energy': None
			})
		else:
			optimal_values[str(W//1000)] = list(map(lambda x: int(round(x, 2)), solution))
			
			# comparison
			optimal_energy = np.sum(np.dot(solution, df['energy_per_sample']))/1000
			optimal_score = np.sum(np.dot(solution, df['score']))/U_MAX
			comparison_data.append({
				'model': '{}-optimal'.format(W//1000),
				'score': optimal_score,
				'energy': optimal_energy
			})
		
	sheets['optimal_values'] = optimal_values
	sheets['comparison'] = pd.DataFrame(comparison_data).set_index('model')
	helpers.append_multiple_sheets(RESULT_FILE, sheets)
	

if __name__ == "__main__":
	W = 55 * 1000
	U_MAX = 3600 * 24 * 1
	report_solution(U_MAX)
