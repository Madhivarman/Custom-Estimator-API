from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf 
import numpy as np 


#define all the parameters
n_filters = 32
filter_width = 2
dilation_rates = [2 ** i for i in range(8)]
pred_steps = 30

filepath = '<path_to_the_timeseries_data>'
orig_df = pd.read_csv(filepath + 'train_dataset.csv')

model_path= '<model_dir_to_save_estimator_model>'

COLUMN_NAMES = orig_df.columns 

"""
 Dataset shape: (13, 731)
 Time series format.
 This code is inspired from this tutorial: https://github.com/JEddy92/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Conv_Intro.ipynb
 
 The above code is written in keras. I took the NN architecture and builting as an Custom Estimator API.
"""

INPUT_COLUMN_NAMES = list(orig_df.columns)[:701]
OUTPUT_COLUMN_NAMES = list(orig_df.columns)[701:]

total_len_inp_features = len(COLUMN_NAMES) - pred_steps

CSV_TYPES= [[0.0]] * len(orig_df.columns)

#read csv file
def csv_input_fn(csv_path, batch_size):
	#input function
	def _input_fn():
		input_filename= tf.train.match_filenames_once(csv_path)

		filename_queue= tf.train.string_input_producer(
			input_filename, num_epochs=None, shuffle=True)

		reader = tf.TextLineReader()
		_, value =  reader.read_up_to(filename_queue, num_records= batch_size)
		value_column = tf.expand_dims(value, -1)

		all_data= tf.decode_csv(value_column, record_defaults=CSV_TYPES)
		inputs= all_data[:len(all_data) - pred_steps]
		labels= all_data[len(all_data) - pred_steps: ]

		inputs= tf.concat(inputs, axis=1)
		labels= tf.concat(labels, axis=1)

		return{'raw_data': inputs}, labels


	return _input_fn


#create an serving input function
def serving_input_fn():

	feature_placeholders={
		'raw_data': tf.placeholder(tf.float32,
			[None, total_len_inp_features])
	}

	features = {
		key: tf.expand_dims(tensor,-1)
		for key, tensor in feature_placeholders.items()
	}

	return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


#create mode function
def cnn_model_fn(features, labels, mode):

	#setup  the mode
	if mode == tf.estimator.ModeKeys.PREDICT:
		tf.logging.info("My Model Function: PREDICT, {}".format(mode))
	elif mode == tf.estimator.ModeKeys.EVAL:
		tf.logging.info("My Model Function: EVAL, {}".format(mode))
	elif mode ==  tf.estimator.ModeKeys.TRAIN:
		tf.logging.info("My Model Function: TRAIN, {}".format(mode))


	#set up the initializer
	#Input layer

	input_layer = tf.reshape(features['raw_data'], [-1, total_len_inp_features,1])

	#conv1 layer
	conv1 = tf.layers.conv1d(input_layer, filters=n_filters,
		kernel_size= filter_width,
		padding='same',
		dilation_rate= dilation_rates[0])

	#conv2 layer
	conv2 = tf.layers.conv1d(conv1, filters=n_filters,
		kernel_size= filter_width,
		padding='same',
		dilation_rate= dilation_rates[1])

	#conv3 layer
	conv3 = tf.layers.conv1d(conv2, filters=n_filters,
		kernel_size= filter_width,
		padding='same',
		dilation_rate= dilation_rates[2])

	#conv4 layer
	conv4 = tf.layers.conv1d(conv3, filters=n_filters,
		kernel_size= filter_width,
		padding='same',
		dilation_rate= dilation_rates[3])

	#conv5 layer
	conv5 = tf.layers.conv1d(conv4, filters=n_filters,
		kernel_size= filter_width,
		padding= 'same',
		dilation_rate= dilation_rates[4])

	#conv6 layer
	conv6 = tf.layers.conv1d(conv5, filters=n_filters,
		kernel_size= filter_width,
		padding='same',
		dilation_rate= dilation_rates[5])

	#conv7 layer
	conv7 = tf.layers.conv1d(conv6, filters=n_filters,
		kernel_size= filter_width,
		padding='same',
		dilation_rate= dilation_rates[6])

	#conv8 layer
	conv8 = tf.layers.conv1d(conv7, filters=n_filters,
		kernel_size= filter_width,
		padding='same',
		dilation_rate= dilation_rates[7])

	#add dense layer
	dense_1 = tf.layers.dense(conv8, units=128,
		activation= tf.nn.relu)

	#add dropout
	dropout= tf.layers.dropout(dense_1, rate=0.4)

	#add dense layer
	dense_2 = tf.layers.dense(dropout, units=1)

	#output layer
	outlen= tf.reshape(dense_2, [-1, len(OUTPUT_COLUMN_NAMES)])

	#predictions
	predictions= tf.layers.dense(outlen, 30, activation= None)


	#write an mode
	if mode == tf.estimator.ModeKeys.PREDICT:

		return tf.estimator.EstimatorSpec(mode= mode, predictions=predictions)

	#calculate the loss
	loss = tf.losses.mean_squared_error(labels, predictions)
	rmse = tf.metrics.root_mean_squared_error(labels, predictions)

	#configure the TrainingOp(for both train and eval modes)
	if mode ==  tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
						loss= loss,
						global_step= tf.train.get_global_step())


		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	#evaluation metrics ops
	eval_metrics_ops = {
		"accuracy": tf.metrics.accuracy(
						labels=labels, predictions= predictions, name='acc-op'),
		"precision": tf.metrics.precision(
						labels=labels, predictions= predictions, name='precision-op'),
		"recall": tf.metrics.recall(
						labels=labels, predictions= predictions, name='recall-op'),
		"auc": tf.metrics.auc(
						labels=labels, predictions= predictions, name='auc-op')
	}

	#write summary scalar to tensorboard
	tf.summary.scalar("Accuracy", eval_metrics_ops['accuracy'][1])
	tf.summary.scalar("Precision", eval_metrics_ops['precision'][1])
	tf.summary.scalar("Recall", eval_metrics_ops['recall'][1])
	tf.summary.scalar("AUC", eval_metrics_ops['auc'][1])


	return tf.estimator.EstimatorSpec(
				mode= mode, loss=loss, eval_metrics_ops= eval_metrics_ops)


#main programs starts here
train_filename= 'train_dataset.csv'
eval_filename= 'eval_dataset.csv'
test_filename= 'test_dataset.csv'

#create an estimator wrapper
estimator= tf.estimator.Estimator(
	model_fn= cnn_model_fn,
	model_dir= model_path)

train_spec= tf.estimator.TrainSpec(
	input_fn= csv_input_fn(filepath + train_filename, 16),
	max_steps= 5000)

exporter = tf.estimator.LatestExporter('Servo', serving_input_fn)

eval_spec = tf.estimator.EvalSpec(
				input_fn= csv_input_fn(
							filepath + eval_filename, 1),
				steps=1,
				exporters= exporter)

#start the training
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
