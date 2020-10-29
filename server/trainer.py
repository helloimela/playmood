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
Example:
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

# Pack features into single NumpyArray
data_features_np = np.array(data_features_df)
print("3 Pack feat into numpy arr ----------------------------")
print(data_features_np)

