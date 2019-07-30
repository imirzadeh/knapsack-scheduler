import sys
from knapsack import helpers
from comet_ml import Experiment
import pandas as pd
from joblib import load
from knapsack.dataset import Dataset
from knapsack.pipeline import Pipeline
from knapsack.nn_utils import run_nn_model
from knapsack.config import get_config_by_id
from knapsack.settings import DATASET_NAME, RETRY_EXPERIMENTS, CURRENT_POOL


def build_model(config):
	p = Pipeline(config)
	p.run_pipeline()
	score = p.validate()
	print("SCORE ==> ", score)
	return score


def measure(config):
	X_test, y_test = Dataset(config.dataset_name, train=False).get()
	model_name = config._get_model_name()
	if 'keras' in model_name:
		for i in range(RETRY_EXPERIMENTS):
			run_nn_model('./models/{}.tflite'.format(config.id), X_test)
	else:
		clf = load('./models/{}.joblib'.format(config.id))
		for i in range(RETRY_EXPERIMENTS):
				for x in X_test:
						clf.predict([x])


def train_models(config_pool=CURRENT_POOL):
		results = []
		num_models = len(config_pool)
		with open("./models/models.txt", "w") as f:
			f.write(str(num_models-1))
		for cfg in config_pool:
			print('running {}'.format(cfg.to_dict()))
			result = cfg.to_dict()
			result['score'] = build_model(cfg)
			results.append(result)
		df = pd.DataFrame(results)
		df = df.set_index('id')
		helpers.create_new_excel_file('report.xlsx', {'models': df})


def read_measurements(dataset_name, config_pool):
	DATASET_SIZE = Dataset(dataset_name, train=False).get()[0].shape[0]
	DATASET_ROUNDS = RETRY_EXPERIMENTS
	IDLE_ENERGY_PER_SECOND = 2.1
	total_samples_calculated = DATASET_SIZE * DATASET_ROUNDS
	
	measurements = {}
	
	for cfg in config_pool:
		energy_counts = []
		duration_counts = []
		with open("./{}.txt".format(cfg.id), 'r') as f:
			lines = list(map(lambda x: float(x.strip()), f.readlines()))
		
		for n, line in enumerate(lines):
			if n % 2 == 0:
				energy_counts.append(line)
			else:
				duration_counts.append(line)
				
		assert len(duration_counts) == len(energy_counts)
		
		average_energy = sum(energy_counts) / len(energy_counts)
		average_time = sum(duration_counts) / len(duration_counts)
		
		total_energy = (average_energy - (average_time * IDLE_ENERGY_PER_SECOND)) * 1000.0
		energy_per_sample = total_energy / total_samples_calculated
		time_per_sample = average_time / total_samples_calculated * 1000.0
		measurements[cfg.id] = {'energy_per_sample': energy_per_sample, 'time_per_sample': time_per_sample}
	return measurements


def generate_report(config_pool=CURRENT_POOL):
	measurements = read_measurements(dataset_name=DATASET_NAME, config_pool=config_pool)
	df = helpers.read_sheet('report.xlsx', 'models')
	new_report = []
	for index, row in df.iterrows():
		row = dict(row)
		measurement = measurements.get(row['id'], None)
		if not measurement:
			continue
		row['energy_per_sample'] = measurement['energy_per_sample']
		row['time_per_sample'] = measurement['time_per_sample']
		new_report.append(row)
	df = pd.DataFrame(new_report).set_index('id')
	helpers.append_sheet('report.xlsx', 'measures', df)

	
if __name__ == "__main__":
	cfg_id = int(sys.argv[1])
	cfg = get_config_by_id(cfg_id)
	print(cfg.to_dict())
	measure(cfg)
