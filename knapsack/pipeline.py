import joblib
from knapsack.dataset import Dataset
from sklearn.metrics import mean_squared_error, accuracy_score
from knapsack.settings import NN_BATCH_SIZE, NN_EPOCHS
from keras.models import load_model


class Pipeline(object):
	def __init__(self, config):
		print(config.to_dict())
		self.id = config.id
		self.model_name = config._get_model_name()
		self.dataset_train = Dataset(config.dataset_name, train=True)
		self.dataset_test = Dataset(config.dataset_name, train=False)
		self.model = config.classifier_model
		self.X_train, self.y_train = self.dataset_train.get()
		self.X_test, self.y_test = self.dataset_test.get()
		self.cached_score = None
		self.graph = None
	
	def run_pipeline(self):
		self.preprocess()
		self.train()
		self.save()
		print(self.model_name)
		
	def preprocess(self):
		pass
	
	def train(self):
		self.model.fit(self.X_train, self.y_train)
		
	def validate(self):
		model_name = self.model_name
		if 'regress' in model_name or 'svr' in model_name or 'lasso' in model_name:
			score = mean_squared_error(self.model.predict(self.X_test), self.y_test)
		elif 'keras' in model_name:
			score = self.model.evaluate(self.id, self.X_test, self.y_test)
		else:
			score = accuracy_score(self.model.predict(self.X_test), self.y_test)
		return score
	
	def save(self):
		if 'keras' in self.model_name:
			self.model.save(self.id)
		else:
			joblib.dump(self.model, './models/{}.joblib'.format(self.id))
		
	
# if __name__ == "__main__":
	# p = Pipeline(CONFIG)
	# p.run_pipeline()
	# print(str(p.model))
	# print(p.validate())
	# # p.model.fit(p.dataset.X, p.dataset.y)
	# # print(p.model.score(p.dataset.X, p.dataset.y))
