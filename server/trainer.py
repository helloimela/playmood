import sys
import pandas as pd
import numpy as np
from util import *
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

'''
Machine learning using CSV as input
Command:
 python trainer.py <dataset>
Example (windows):
 python trainer.py set1.csv
'''

# Fetch input data from args, convert to pandas dataframe. Use UTF-8
DATA_DIR = Path("datasets/")
DATA_FILE = DATA_DIR / sys.argv[1]
#TEST_FILE = DATA_DIR / sys.argv[2]
data_df = pd.read_csv(DATA_FILE)
print("1 Input data in dataframe ----------------------------")
print(data_df.head())

# Separate features and labels # todo automatically detect features and labels
data_features_df = data_df.copy()
data_labels_df = data_df.pop('mood')
print("2 Separate features and label ----------------------------")
print(data_features_df.head())
print(data_labels_df.head())

# ==================================
print("3 Create input model ----------------------------")
# Create a symbolic input
input = tf.keras.Input(shape=(), dtype=tf.float32)

# Do a calculation using is
result = 2*input + 1

# the result doesn't have a value
#result
calc = tf.keras.Model(inputs=input, outputs=result)

# To build the preprocessing model, start by building a set of symbolic keras.Input objects, matching the names and data-types of the CSV columns.
inputs = {}

for name, column in data_features_df.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

print(inputs)


preprocessed_inputs = []
print("4a ----------------------------")
# Skip numeric input
print("4b ----------------------------")
# String input
for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    lookup = preprocessing.StringLookup(vocabulary=np.unique(data_features_df[name]))
    one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

print("5 ----------------------------")
# Concatenate all preprocessed inputs
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
data_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
#tf.keras.utils.plot_model(model = data_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

print("6 ----------------------------")
data_features_dict = {name: np.array(value)
                         for name, value in data_features_df.items()}
features_dict = {name:values[:1] for name, values in data_features_dict.items()}
print(data_preprocessing(features_dict))

print("7 ----------------------------")
#Now build the model on top of this:

def training_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.optimizers.Adam())
  return model

training_model = training_model(data_preprocessing, inputs)

#When you train the model, pass the dictionary of features as x, and the label as y.

training_model.fit(x=data_features_dict, y=data_labels_df, epochs=10)

# Test
#test_loss, test_accuracy = norm_training_model.evaluate(data_features_np, verbose=0)
#print('Testing finished. loss={} accuracy={}'.format(test_loss, test_accuracy))
