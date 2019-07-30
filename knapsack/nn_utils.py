import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from joblib import load
import tensorflow as tf
from tensorflow.python import keras
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Conv1D, MaxPool1D, MaxPooling1D
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K


class KerasModel(object):
	def __init__(self, model, train_epochs=10, batch_size=32, quantized=False, prune_params=None):
		self.model = model
		self.graph = tf.Graph()
		self.session = tf.Session(graph=self.graph)
		self.train_epochs = train_epochs
		self.batch_size = batch_size
		self.quantized = quantized
		self.prune_params = prune_params
		
	def fit(self, X_train, y_train):
		with self.graph.as_default(), self.session.as_default():
			callbacks = None
			if self.prune_params:
				callbacks = [
					pruning_callbacks.UpdatePruningStep(),
				]
			self.model.fit(X_train, y_train, epochs=self.train_epochs, batch_size=self.batch_size, callbacks=callbacks)
	
	def save(self, id):
		with self.graph.as_default(), self.session.as_default():
			keras_file_path = './models/{}.h5'.format(id)
			print(self.model.summary())
			print("**"*20)
			if self.prune_params:
				self.model = prune.strip_pruning(self.model)
			print(self.model.summary())
			save_keras_model(self.model, keras_file_path)
		tflite_path = './models/{}.tflite'.format(id)
		convert_keras_file_to_tflite(keras_file_path, tflite_path, quantized=self.quantized)
	
	def evaluate(self, id, X_test, y_test):
		# with self.graph.as_default(), self.session.as_default():
		# 	model = load_model(filepath='./models/{}.h5'.format(id))
		# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		# 	_, score = model.evaluate(X_test, y_test, batch_size=self.batch_size)
		# 	return score
		score = eval_nn_model('./models/{}.tflite'.format(id), X_test, y_test)
		return score
	
	def __str__(self):
		model_type = None
		first_layer = str(self.model.layers[0]).lower()
		if 'convolutional' in first_layer or 'conv' in first_layer:
			model_type = 'CNN'
		elif 'dense' in first_layer:
			model_type = 'MLP'
		else:
			model_type = 'RNN'
		clf_name = 'keras-{}'.format(model_type)
		return clf_name

	
def get_mlp_diabetes(size=20):
	keras_model = KerasModel(model=None, train_epochs=80)
	graph, session = keras_model.graph, keras_model.session
	print(graph, session)
	with graph.as_default():
		with session.as_default():
			model = tf.keras.Sequential()
			model.add(Dense(size, activation='relu', input_shape=(10,)))
			model.add(Dense(size, activation='relu'))
			model.add(Dropout(rate=0.5))
			model.add(Dense(1, activation='relu'))
			model.compile(loss='mse', optimizer='adam', metrics=['mse'])
			keras_model.model = model
			return keras_model
			
	
def get_time_series_cnn_model(n_timesteps, n_features, n_outputs,
							  conv_layers=[(64, 3), (64, 3)], dense_size=100,
							  quantized=False, prune_params=None):
	keras_model = KerasModel(model=None, quantized=quantized, prune_params=prune_params)
	graph, session = keras_model.graph, keras_model.session
	print(graph, session)
	with graph.as_default():
		with session.as_default():
			model = tf.keras.Sequential()
			filters, kernel_size = conv_layers[0]
			model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(n_timesteps,n_features)))
			if len(conv_layers) > 1:
				filters, kernel_size = conv_layers[1]
				model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
			model.add(Dropout(0.5))
			model.add(MaxPooling1D(pool_size=2))
			if len(conv_layers) > 2:
				filters, kernel_size = conv_layers[2]
				model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(n_timesteps,n_features)))
				model.add(MaxPooling1D(pool_size=2))
			model.add(Flatten())
			model.add(Dense(dense_size, activation='relu'))
			model.add(Dense(n_outputs, activation='softmax'))
			if prune_params:
				model = prune.prune_low_magnitude(model, **prune_params)
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
			keras_model.model = model
			return keras_model
			

def save_keras_model(model, file_path):
	model.save(file_path, include_optimizer=True)

	
def representative_HAR_gen():
	data = load('./datasets/{}.{}'.format('UCI_HAR_NN', 'train'))
	X = data['X']
	for inp in X:
		inp = np.array(inp.reshape(1, 128, 9), dtype=np.float32)
		yield [inp]
		

def convert_keras_file_to_tflite(keras_model_path, tf_lite_path, quantized=False):
	converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model_path)
	if quantized:
		converter.representative_dataset = representative_HAR_gen
		converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
	tflite_quant_model = converter.convert()
	open(tf_lite_path, "wb").write(tflite_quant_model)


def eval_nn_model(tf_lite_path, X_test, y_test):
	with tf.device("/cpu:0"):
		interpreter = tf.lite.Interpreter(model_path=tf_lite_path)
		interpreter.allocate_tensors()
		
		# Get input and output tensors.
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		predictions = []
		
		for x in X_test[np.random.randint(X_test.shape[0], size=50)]:
			x = np.array(x.reshape(1, 128, 9), dtype=np.float32)
			input_shape = input_details[0]['shape']
			input_data = x  # np.array(np.random.random_sample(input_shape), dtype=np.float32)
			interpreter.set_tensor(input_details[0]['index'], input_data)
			
			interpreter.invoke()
			
			# The function `get_tensor()` returns a copy of the tensor data.
			# Use `tensor()` in order to get a pointer to the tensor.
			output_data = interpreter.get_tensor(output_details[0]['index'])
			predictions.append(output_data[0])
		
		correct = 0
		for idx, pred in enumerate(predictions):
			pred = np.argmax(pred)
			target = np.argmax(y_test[idx])
			if pred == target:
				correct += 1
		accuracy = correct * 1.0 / len(y_test)
		return accuracy
	

def run_nn_model(tf_lite_path, X_test):
	with tf.device("/cpu:0"):
		interpreter = tf.lite.Interpreter(model_path=tf_lite_path)
		interpreter.allocate_tensors()
		
		# Get input and output tensors.
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		
		for x in X_test:
			x = np.array(x.reshape(1, 128, 9), dtype=np.float32)
			input_data = x  # np.array(np.random.random_sample(input_shape), dtype=np.float32)
			interpreter.set_tensor(input_details[0]['index'], input_data)
			
			interpreter.invoke()
			
			# The function `get_tensor()` returns a copy of the tensor data.
			# Use `tensor()` in order to get a pointer to the tensor.
			output_data = interpreter.get_tensor(output_details[0]['index'])
