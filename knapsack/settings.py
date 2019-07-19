import tensorflow as tf

# comet-ml API keys
COMET_ML_KEY = dict(api_key="1UNrcJdirU9MEY0RC3UCU7eAg", project_name="trial-1", workspace="knapsack")

# name of the dataset we are working
DATASET_NAME = 'UCI_HAR'

# number of retry for experiments
from knapsack.config import CONFIG_POOL_REG, CONFIG_POOL_NN
RETRY_EXPERIMENTS = 2
CURRENT_POOL = CONFIG_POOL_NN

# output excel file
RESULT_FILE = './report.xlsx'

# Neural net settings
NN_EPOCHS = 3
NN_BATCH_SIZE = 32

