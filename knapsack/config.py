from enum import Enum
from sklearn.datasets import make_classification
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.feature_selection import RFE, SelectKBest, chi2, mutual_info_classif
from knapsack.settings import DATASET_NAME
from knapsack.nn_utils import get_time_series_cnn_model, get_mlp_diabetes
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule


class Config(object):
	def __init__(self, id=None, dataset_name=None, classifier_model=None):
		self.id = id
		self.dataset_name = dataset_name
		self.classifier_model = classifier_model
		if 'keras' in self._get_model_name():
			self.dataset_name = self.dataset_name + "_NN"
	
	def _get_model_name(self):
		clf_name = self.classifier_model.__str__()
		if clf_name.startswith("RFE"):
			clf_name = clf_name[4:]  # remove RFE(...)
			clf_name = "RFE({})".format(clf_name[:clf_name.find("(")])
		elif 'keras' in clf_name or 'Keras' in clf_name:
			clf_name = clf_name
		else:
			clf_name = clf_name[:clf_name.find("(")]
		return clf_name
	
	def to_dict(self):
		return {
			'id': str(self.id),
			'dataset': self.dataset_name,
			'model': self._get_model_name(),
		}



MODEL_POOL = [
	# Complete Models
	SVC(C=0.1, gamma='scale'),
	DecisionTreeClassifier(criterion='entropy', max_depth=10),
	DecisionTreeClassifier(criterion='gini', max_depth=10),
	DecisionTreeClassifier(criterion='entropy', max_depth=15),
	DecisionTreeClassifier(criterion='gini', max_depth=15),
	KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree'),
	KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree'),
	GaussianNB(),
	LogisticRegression(solver='lbfgs'),
	RandomForestClassifier(n_estimators=10, criterion='entropy'),
	RandomForestClassifier(n_estimators=20, criterion='entropy'),
	RandomForestClassifier(n_estimators=10, criterion='gini'),
	RandomForestClassifier(n_estimators=20, criterion='gini'),
	AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=10),
	AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), n_estimators=10),
	AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), n_estimators=20),
	GradientBoostingClassifier(n_estimators=10, max_depth=5),
	GradientBoostingClassifier(n_estimators=10, max_depth=10),
	GradientBoostingClassifier(n_estimators=20, max_depth=10),
	
	# Feature Selection
	RFE(estimator=SVC(kernel='linear',C=0.5, gamma='scale'), n_features_to_select=5),
	RFE(estimator=DecisionTreeClassifier(criterion='entropy', max_depth=10), n_features_to_select=5),
]

CONFIG_POOL = [Config(id=i, dataset_name='artificial_clf_multi', classifier_model=clf) for i, clf in enumerate(MODEL_POOL)]

MODEL_POOL_REG = [
	SVR(degree=3, C=0.5, gamma='scale'),
	DecisionTreeRegressor(criterion='mse', max_depth=10),
	KNeighborsRegressor(n_neighbors=3, algorithm='ball_tree'),
	RandomForestRegressor(n_estimators=20, criterion='mse', max_depth=5),
	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=3, criterion='mse'), n_estimators=10),
	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=3, criterion='mse'), n_estimators=30),
	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=5, criterion='mse'), n_estimators=30),
	GradientBoostingRegressor(n_estimators=10),
	GradientBoostingRegressor(n_estimators=15),
	get_mlp_diabetes(25),
	get_mlp_diabetes(50),
	get_mlp_diabetes(100),
	
]


CONFIG_POOL_REG = [Config(id=i, dataset_name=DATASET_NAME, classifier_model=clf) for i, clf in enumerate(MODEL_POOL_REG)]


prune_poly_90 = {
      'pruning_schedule': pruning_schedule.PolynomialDecay(initial_sparsity=0.10,
                                                   final_sparsity=0.90,
                                                   begin_step=0,
												   end_step = 920,
                                                   frequency=50)
}


prune_poly_75 = {
	'pruning_schedule': pruning_schedule.PolynomialDecay(initial_sparsity=0.50,
														 final_sparsity=0.75,
														 begin_step=100,
														 end_step=920,
														 frequency=10)
}


prune_const_90 = {
	'pruning_schedule': pruning_schedule.ConstantSparsity(target_sparsity=0.90,
														 begin_step=50,
														 end_step=-1,
														 frequency=50)
}


prune_const_75 = {
	'pruning_schedule': pruning_schedule.ConstantSparsity(target_sparsity=0.75,
														 begin_step=100,
														 end_step=920,
														 frequency=50)
}


NN_MODELS_POOL = [
	get_time_series_cnn_model(128, 9, 6, [(32, 3), (64, 3), (128, 3)], 50),
	get_time_series_cnn_model(128, 9, 6, [(32, 3), (64, 3), (128, 3)], 50, quantized=True),
	get_time_series_cnn_model(128, 9, 6, [(32, 3), (64, 3), (128, 3)], 50, prune_params=prune_const_90),
	get_time_series_cnn_model(128, 9, 6, [(64, 3), (64, 3)], 100, prune_params=prune_const_90, quantized=True),
	SVC(kernel='rbf', C=5, gamma='scale'),
	# AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5, criterion='entropy'), n_estimators=50),
	KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree'),
	GaussianNB(),
	RandomForestClassifier(n_estimators=10, criterion='entropy'),
	RandomForestClassifier(n_estimators=15, criterion='entropy'),
]

CONFIG_POOL_NN = [Config(id=i, dataset_name='UCI_HAR', classifier_model=clf) for i, clf in enumerate(NN_MODELS_POOL)]


# TODO Fix the static config pool
def get_config_by_id(config_id):
	for cfg in CONFIG_POOL_NN:
		if cfg.id == config_id:
			return cfg
