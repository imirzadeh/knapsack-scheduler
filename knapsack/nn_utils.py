import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model
import tensorflow as tf
from keras import backend as K


class KerasModel(object):
	def __init__(self, model,train_epochs=5, batch_size=32):
		self.model = model
		self.graph = tf.Graph()
		self.session = tf.Session(graph=self.graph)
		self.train_epochs = train_epochs
		self.batch_size = batch_size
		
	def fit(self, X_train, y_train):
		with self.graph.as_default(), self.session.as_default():
			self.model.fit(X_train, y_train, epochs=self.train_epochs, batch_size=self.batch_size)
	
	def save(self, id):
		with self.graph.as_default(), self.session.as_default():
			keras_file_path = './models/{}.h5'.format(id)
			save_keras_model(self.model, keras_file_path)
		tflite_path = './models/{}.tflite'.format(id)
		convert_keras_file_to_tflite(keras_file_path, tflite_path)
	
	def evaluate(self, id, X_test, y_test):
		with self.graph.as_default(), self.session.as_default():
			model = load_model(filepath='./models/{}.h5'.format(id))
			_, score = model.evaluate(X_test, y_test, batch_size=self.batch_size)
			return score
	
	def __str__(self):
		model_type = None
		first_layer = str(self.model.layers[0]).lower()
		if 'convolutional' in first_layer or 'Conv' in first_layer:
			model_type = 'CNN'
		elif 'dense' in first_layer:
			model_type = 'MLP'
		else:
			model_type = 'RNN'
		clf_name = 'keras-{}'.format(model_type)
		return clf_name
	
	
def get_time_series_cnn_model(n_timesteps, n_features, n_outputs, conv_layers=[(64, 3), (64, 3)], dense_size=100):
	keras_model = KerasModel(model=None)
	graph, session = keras_model.graph, keras_model.session
	print(graph, session)
	with graph.as_default():
		with session.as_default():
			model = Sequential()
			filters, kernel_size = conv_layers[0]
			model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(n_timesteps,n_features)))
			if len(conv_layers) > 1:
				filters, kernel_size = conv_layers[1]
				model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
			model.add(Dropout(0.5))
			model.add(MaxPooling1D(pool_size=2))
			model.add(Flatten())
			model.add(Dense(dense_size, activation='relu'))
			model.add(Dense(n_outputs, activation='softmax'))
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
			keras_model.model = model
			return keras_model
			

def save_keras_model(model, file_path):
	model.save(file_path, include_optimizer=True)
	

def convert_keras_file_to_tflite(keras_model_path, tf_lite_path):
	converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model_path)
	# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
	tflite_quant_model = converter.convert()
	open(tf_lite_path, "wb").write(tflite_quant_model)


def run_nn_model(tf_lite_path, X_test):
	interpreter = tf.lite.Interpreter(model_path=tf_lite_path)
	interpreter.allocate_tensors()
	
	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	
	for x in X_test:
		x = np.array(x.reshape(1, 128, 9), dtype=np.float32)
		input_shape = input_details[0]['shape']
		input_data = x  # np.array(np.random.random_sample(input_shape), dtype=np.float32)
		interpreter.set_tensor(input_details[0]['index'], input_data)
		
		interpreter.invoke()
		
		# The function `get_tensor()` returns a copy of the tensor data.
		# Use `tensor()` in order to get a pointer to the tensor.
		output_data = interpreter.get_tensor(output_details[0]['index'])
