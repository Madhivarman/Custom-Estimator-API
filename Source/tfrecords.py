import tensorflow as tf
import pandas as pd
import numpy as np

df = pd.read_csv("F:/work projects/living_spaces_projects/presenting/TimeSeriesData/train_dataset.csv")


COLUMN_NAMES = df.columns
INPUT_COLUMN_NAMES = list(df.columns)[:671]
OUTPUT_COLUMN_NAMES = list(df.columns)[671:]
pred_steps = 30
total_len_inp_features = len(COLUMN_NAMES) - pred_steps

CSV_TYPE = [[0.0]] * len(INPUT_COLUMN_NAMES) + [[0.0]] * len(OUTPUT_COLUMN_NAMES)

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

def csv_input_fn(csv_path, batch_size):
    
    dataset = tf.data.TextLineDataset(csv_path)
    #parse each line
    dataset = dataset.map(_parse_line)
    #shuffle, repeat, and batch the example
    dataset = dataset.shuffle(100).repeat().batch(batch_size)

    return dataset

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


tfrecords = read_dataset('F:/work projects/living_spaces_projects/presenting/TimeSeriesData/train_dataset.csv', 4)

print("Data Passed as Tensor Format")
print("-" * 100)
print(tfrecords)

