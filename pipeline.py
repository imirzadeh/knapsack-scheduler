import joblib
from dataset import Dataset
from sklearn.metrics import accuracy_score, mean_squared_error


class Pipeline(object):
	def __init__(self, config):
		self.id = config.id
		self.dataset_train = Dataset(config.dataset_name, train=True)
		self.dataset_test = Dataset(config.dataset_name, train=False)
		self.clf = config.classifier_model
		self.X_train, self.y_train = self.dataset_train.get()
		self.X_test, self.y_test = self.dataset_test.get()
	
	def run_pipeline(self):
		self.preprocess()
		self.train()
		self.save()
		print(self.clf.__str__()[:self.clf.__str__().find('(')])
		
	def preprocess(self):
		pass
	
	def train(self):
		self.clf.fit(self.X_train, self.y_train)
		
	def validate(self):
		model_name = self.clf.__str__().lower()
		score = mean_squared_error(self.clf.predict(self.X_test), self.y_test)
		# if 'regress' in model_name or 'svr' in model_name:
		# 	score = mean_squared_error(self.clf.predict(self.X_test), self.y_test)
		# else:
		# 	score = accuracy_score(self.clf.predict(self.X_test), self.y_test)
		return score
	
	def save(self):
		joblib.dump(self.clf, './models/{}.joblib'.format(self.id))
		
	
# if __name__ == "__main__":
	# p = Pipeline(CONFIG)
	# p.run_pipeline()
	# print(str(p.clf))
	# print(p.validate())
	# # p.clf.fit(p.dataset.X, p.dataset.y)
	# # print(p.clf.score(p.dataset.X, p.dataset.y))
