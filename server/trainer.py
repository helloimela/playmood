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
print(data_df.dtypes)

print("2 Convert string to numeric")
data_df['location'] = pd.Categorical(data_df['location'])
data_df['tempfeel'] = pd.Categorical(data_df['tempfeel'])
data_df['time'] = pd.Categorical(data_df['time'])
data_df['mood'] = pd.Categorical(data_df['mood'])

data_df['location'] = data_df.location.cat.codes
data_df['tempfeel'] = data_df.tempfeel.cat.codes
data_df['time'] = data_df.time.cat.codes
data_df['mood'] = data_df.mood.cat.codes

print(data_df.head())
print(data_df.dtypes)

# Separate features and labels # todo automatically detect features and labels
print("3 Separate features and label ----------------------------")
data_features_df = data_df.copy()
data_labels_df = data_df.pop('mood')
print(data_features_df.head())
print(data_labels_df.head())

print("4 Load data ----------------------------")
dataset = tf.data.Dataset.from_tensor_slices((data_df.values, data_labels_df.values))

for feat, targ in dataset.take(10):
  print ('Features: {}, Target: {}'.format(feat, targ))
tf.constant(data_df['location'])
tf.constant(data_df['tempfeel'])
tf.constant(data_df['time'])

train_dataset = dataset.shuffle(len(data_df)).batch(1)

print("5 TRAIN----------------------------")
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)


print("6 TEST ----------------------------")
# Test
test_loss, test_accuracy = model.evaluate(dataset.batch(1), verbose=0)
print('Testing finished. loss={} accuracy={}'.format(test_loss, test_accuracy))
