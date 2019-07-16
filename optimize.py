import cvxpy as cp
import numpy as np
import pandas as pd


def read_measurements_file():
	df = pd.read_csv('./measures.csv')
	return df


def is_regression_dataset(dataset_name):
	dataset_name = dataset_name.lower()
	regression_dataset_identifiers = ['california', 'boston', 'housing', 'regression', 'regress', 'reg']
	for id in regression_dataset_identifiers:
		if id in dataset_name:
			return True
	return False
	

def scale_values(values):
	max_val = max(values)
	# scale all from zero to one
	values = [(v * 1.0)/max_val for v in values]
	return values


def one_minus_x(values):
	assert max(values) <= 1.0 and min(values) >= 0
	values = [1.0-v for v in values]
	return values

	
def prepare_optimization_problem_variables():
	df = read_measurements_file()
	ids = list(df['id'])
	weights = list(df['energy_per_sample'])
	values = list(df['score'])
	dataset_name = get_dataset_name(df)
	
	if is_regression_dataset(dataset_name):
		values = one_minus_x(scale_values(values))
	
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
	is_regression = is_regression_dataset(get_dataset_name(df))
	
	min_energy = U_MAX * min(list(df['energy_per_sample'])) + 1100
	max_energy = U_MAX * max(list(df['energy_per_sample']))
	NUM_W = 20
	delta = (max_energy - min_energy) // NUM_W
	W_list = [((min_energy + i * delta)//1000)*1000 for i in range(NUM_W)]
	# W_list = [50000, 75000, 100000, 150000, 200000]
	print('min energy: {}, max energy: {}'.format(min_energy//1000, max_energy//1000))
	print(W_list)
	
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
		optimal_values[str(W//1000)] = list(map(lambda x: int(round(x, 2)), solution))
		
		# comparison
		optimal_energy = np.sum(np.dot(solution, df['energy_per_sample']))/1000
		optimal_score = np.sum(np.dot(solution, df['score']))/U_MAX
		comparison_data.append({
			'model': '{}-optimal'.format(W//1000),
			'score': optimal_score,
			'energy': optimal_energy
		})
		
		# new_report = []
		# for index, row in df.iterrows():
		# 	row = dict(row)
		# 	row['solution'] = int(round(solution[ids.index(index)]))
		# 	row['optimal_energy'] = row['solution'] * row['energy_per_sample']
		# 	row['non_optimal_energy'] = U_MAX * row['energy_per_sample']
		# 	new_report.append(row)
	
	sheets['optimal_values'] = optimal_values
	sheets['comparison'] = pd.DataFrame(comparison_data).set_index('model')
	
	excel_writer = pd.ExcelWriter('report.xlsx')
	for sheet_name in sheets.keys():
		sheets[sheet_name].to_excel(excel_writer, sheet_name)
	excel_writer.save()

# df = pd.DataFrame(new_report).set_index('id')
	# df.to_csv('./solution.csv')
	
if __name__ == "__main__":
	W = 55 * 1000
	U_MAX = 3600 * 24 * 1
	report_solution(U_MAX)
