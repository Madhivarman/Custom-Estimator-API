import tensorflow as tf
import pandas as pd
import numpy as np

df = pd.read_csv("FILENAME.csv")

#Data Shape is (14, 731)
"""
    For Time Series, first 671 data point is used for Training and last 30 data point is taken
    as target.
    
    Problem Statement: 
        Predicting Store Traffic for 30 days.
"""
COLUMN_NAMES = df.columns
INPUT_COLUMN_NAMES = list(df.columns)[:671] #change the index range according to your data
OUTPUT_COLUMN_NAMES = list(df.columns)[671:]
pred_steps = 30 #days you want the prediction
total_len_inp_features = len(COLUMN_NAMES) - pred_steps

CSV_TYPE = [[0.0]] * len(INPUT_COLUMN_NAMES) + [[0.0]] * len(OUTPUT_COLUMN_NAMES) #all data points as float

def _parse_line(line):
    
    #decode csv
    fields = tf.decode_csv(line, record_defaults= CSV_TYPE)
    #get the input features
    features = df.iloc[:, :total_len_inp_features].values
    #get the labels features
    labels = df.iloc[:, total_len_inp_features:].values

    #from the list of tensors to tensor
    inputs = tf.concat(features, axis=1)
    label = tf.concat(labels, axis=1)

    return {'raw_data': inputs}, label

#convert csv_file into Tensors
def csv_input_fn(csv_path, batch_size):
    
    dataset = tf.data.TextLineDataset(csv_path).skip(1) #skip the first line in the csv_file
    #parse each line
    dataset = dataset.map(_parse_line)
    #shuffle, repeat, and batch the example
    dataset = dataset.shuffle(100).repeat().batch(batch_size)

    return dataset

#reading the dataset and converting into tensors
def read_dataset(filename, batch_size):
    
    input_filename = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(
            input_filename, num_epochs=None, shuffle=True)

    reader =  tf.TextLineReader()
    _, value = reader.read_up_to(filename_queue, num_records= batch_size)
    value_column = tf.expand_dims(value, -1)

    #all data are list of tensors
    all_data = tf.decode_csv(value_column, record_defaults= CSV_TYPE)
    inputs = all_data[:len(all_data)-pred_steps]
    label = all_data[len(all_data)-pred_steps:]

    #from list of tensors to tensor with one more dimension
    inputs = tf.concat(inputs, axis=1)
    label = tf.concat(label, axis=1)

    return {'raw-data': inputs}, label


#converted Time Series data into TF records to train a Simple Conv1D Model
tfrecords = read_dataset('FILENAME.csv', 4)

