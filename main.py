import sys
from comet_ml import Experiment
import pandas as pd
from joblib import load
from dataset import Dataset
from pipeline import Pipeline
from credentials import COMET_ML_KEY
from config import CONFIG_POOL_REG, get_config_by_id

RETRY_EXPERIMENTS = 10


def run(config):
	p = Pipeline(config)
	p.run_pipeline()
	score = p.validate()
	return score


def measure(config):
	X_test, y_test = Dataset(config.dataset_name, train=False).get()
	clf = load('./models/{}.joblib'.format(config.id))
	for i in range(RETRY_EXPERIMENTS):
		for x in X_test:
			clf.predict([x])


def create_comet_experiment():
	exp = Experiment(**COMET_ML_KEY)
	return exp


def train_models(config_pool):
	# experiment = create_comet_experiment()
	results = []
	
	for cfg in config_pool:
		print('running {}'.format(cfg.to_dict()))
		result = cfg.to_dict()
		result['val_accuracy'] = run(cfg)
		results.append(result)
	df = pd.DataFrame(results)
	df = df.set_index('id')
	df.to_csv('results.csv')
	# experiment.log_asset_folder('./data')
	# experiment.log_asset_folder('./models')
	# experiment.log_asset('./results.csv')


def read_measurements(dataset_name, config_pool):
	DATASET_SIZE = Dataset(dataset_name, train=False).get()[0].shape[0]
	DATASET_ROUNDS = RETRY_EXPERIMENTS
	IDLE_ENERGY_PER_SECOND = 2.1
	total_samples_calculated = DATASET_SIZE * DATASET_ROUNDS
	
	measurements = {}
	
	for cfg in config_pool:
		energy_counts = []
		duration_counts = []
		with open("{}.txt".format(cfg.id), 'r') as f:
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


def generate_report(dataset_name, config_pool):
	measurements = read_measurements(dataset_name=dataset_name, config_pool=config_pool)
	df = pd.read_csv('./results.csv')
	new_report = []
	for index, row in df.iterrows():
		row = dict(row)
		measurement = measurements.get(row['id'], None)
		if not measurement:
			continue
		row['val_accuracy'] = row['val_accuracy'] * 100
		row['energy_per_sample'] = measurement['energy_per_sample']
		row['time_per_sample'] = measurement['time_per_sample']
		new_report.append(row)
	df = pd.DataFrame(new_report).set_index('id')
	df.to_csv('./measures.csv')

	
if __name__ == "__main__":
	# train_models()
	# generate_report('california_hosuing', CONFIG_POOL_REG)
	
	cfg_id = int(sys.argv[1])
	cfg = get_config_by_id(cfg_id)
	print(cfg.id)
	measure(cfg)