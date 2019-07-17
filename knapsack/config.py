from enum import Enum
from sklearn.datasets import make_classification
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.feature_selection import RFE, SelectKBest, chi2, mutual_info_classif


class Config(object):
	def __init__(self, id=None, dataset_name=None, classifier_model=None):
		self.id = id
		self.dataset_name = dataset_name
		self.classifier_model = classifier_model
	
	def _get_model_name(self):
		clf_name = self.classifier_model.__str__()
		if clf_name.startswith("RFE"):
			clf_name = clf_name[4:]  # remove RFE(...)
			clf_name = "RFE({})".format(clf_name[:clf_name.find("(")])
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

# MODEL_POOL_REG = [
# 	SVR(degree=2, C=0.2, gamma='scale'),
# 	SVR(degree=3, C=0.2, gamma='scale'),
# 	SVR(degree=5, C=0.2, gamma='scale'),
# 	DecisionTreeRegressor(criterion='mse', max_depth=4),
# 	DecisionTreeRegressor(criterion='mse', max_depth=8),
# 	DecisionTreeRegressor(criterion='mse', max_depth=10),
# 	KNeighborsRegressor(n_neighbors=3, algorithm='ball_tree'),
# 	KNeighborsRegressor(n_neighbors=3, algorithm='kd_tree'),
# 	LinearRegression(),
# 	RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=4),
# 	RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=8),
# 	RandomForestRegressor(n_estimators=20, criterion='mse'),
# 	RandomForestRegressor(n_estimators=20, criterion='mse', max_depth=4),
# 	RandomForestRegressor(n_estimators=30, criterion='mse'),
# 	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=3, criterion='mse'), n_estimators=10),
# 	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=3, criterion='mse'), n_estimators=20),
# 	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=5, criterion='mse'), n_estimators=10),
# 	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=5, criterion='mse'), n_estimators=20),
# 	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=8, criterion='mse'), n_estimators=20),
# 	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=8, criterion='mse'), n_estimators=50),
# 	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=10, criterion='mse'), n_estimators=20),
# 	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=10, criterion='mse'), n_estimators=50),
# 	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=20, criterion='mse'), n_estimators=10),
# 	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=20, criterion='mse'), n_estimators=20),
# 	GradientBoostingRegressor(n_estimators=10),
# 	GradientBoostingRegressor(n_estimators=15),
# 	GradientBoostingRegressor(n_estimators=20),
# 	GradientBoostingRegressor(n_estimators=30),
# ]

MODEL_POOL_REG = [
	SVR(degree=2, C=0.2, gamma='scale'),
	DecisionTreeRegressor(criterion='mse', max_depth=4),
	RandomForestRegressor(n_estimators=30, criterion='mse'),
	AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=20, criterion='mse'), n_estimators=20),
	GradientBoostingRegressor(n_estimators=30),
]


CONFIG_POOL_REG = [Config(id=i, dataset_name='california_hosuing', classifier_model=clf) for i, clf in enumerate(MODEL_POOL_REG)]


def get_config_by_id(config_id):
	for cfg in CONFIG_POOL_REG:
		if cfg.id == config_id:
			return cfg