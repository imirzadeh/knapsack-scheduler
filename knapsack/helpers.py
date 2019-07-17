import pandas as pd
from openpyxl import load_workbook
from comet_ml import Experiment
from knapsack.settings import COMET_ML_KEY, RESULT_FILE


def append_sheet(excel_file, sheet_name, df):
	book = load_workbook(excel_file)
	with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
		writer.book = book
		writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
		df.to_excel(writer, sheet_name)
		writer.save()


def append_multiple_sheets(excel_file, sheets):
	for sheet_name in sheets.keys():
		df = sheets[sheet_name]
		append_sheet(excel_file, sheet_name, df)


def read_sheet(excel_file, sheet_name):
	df = pd.read_excel(excel_file, sheet_name=sheet_name)
	return df


def create_new_excel_file(excel_file, sheets):
	excel_writer = pd.ExcelWriter(excel_file)
	for sheet_name in sheets.keys():
		sheets[sheet_name].to_excel(excel_writer, sheet_name)
	excel_writer.save()


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


def upload_experiment():
	experiment = Experiment(**COMET_ML_KEY)
	experiment.log_asset_folder('./datasets')
	experiment.log_asset_folder('./models')
	experiment.log_asset_folder('./knapsack')
	experiment.log_asset(RESULT_FILE)
