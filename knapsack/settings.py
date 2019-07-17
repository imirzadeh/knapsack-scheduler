
# comet-ml API keys
COMET_ML_KEY = dict(api_key="1UNrcJdirU9MEY0RC3UCU7eAg", project_name="trial-1", workspace="knapsack")

# name of the dataset we are working
DATASET_NAME = 'diabetes_regression'

# number of retry for experiments
from knapsack.config import CONFIG_POOL_REG
RETRY_EXPERIMENTS = 10
CURRENT_POOL = CONFIG_POOL_REG

# output excel file
RESULT_FILE = './report.xlsx'