from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston, fetch_california_housing


class Dataset(object):
	def __init__(self, dataset_name, train=True):
		extension = 'train' if train else 'test'
		data = load('./data/{}.{}'.format(dataset_name, extension))
		self.X, self.y = data['X'], data['Y']
	
	def get(self):
		return self.X, self.y


def make_artificial_clf_binary():
	name = 'artificial_clf_binary'
	X, y = make_classification(n_samples=1000, n_classes=2, n_informative=10, n_features=20)
	process_dataset(name, X, y)


def make_artificial_clf_multi():
	name = 'artificial_clf_multi'
	X, y = make_classification(n_samples=1000, n_classes=5, n_features=20, n_informative=10)
	process_dataset(name, X, y)
	

def make_regression_1():
	name = 'regression_1'
	X, y = make_regression(n_samples=1000, n_features=50, n_informative=10, noise=0.15)
	process_dataset(name, X, y)

def make_boston():
	name = 'california_hosuing'
	X, y = fetch_california_housing(return_X_y=True)
	process_dataset(name, X, y)
	
def process_dataset(name, X, y, supervised=True):
	if supervised:
		X = StandardScaler().fit_transform(X)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
		dump({'X': X_train, 'Y': y_train}, './data/{}.train'.format(name))
		dump({'X': X_test, 'Y': y_test}, './data/{}.test'.format(name))
	else:
		raise Exception("not supporting unspervised datasets yet!")

	
if __name__ == "__main__":
	# make_artificial_clf_binary()
	# make_artificial_clf_multi()
	# make_regression_1()
	make_boston()
# 	dataset_name  = 'moons_0'
# 	X, y = make_moons(n_samples=80, noise=0.2)
#	X = StandardScaler().fit_transform(X)
# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# 	dump({'X': X_train, 'Y': y_train}, './data/{}.train'.format(dataset_name))
# 	dump({'X': X_test, 'Y': y_test}, './data/{}.test'.format(dataset_name))