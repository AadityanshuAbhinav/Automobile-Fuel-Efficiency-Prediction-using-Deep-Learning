import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 
import tensorflow as tf 
from tensorflow import keras 
from keras import layers 
import warnings 

warnings.filterwarnings('ignore')

# Load and inspect dataset
df = pd.read_csv('auto-mpg.csv') 
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# Clean data
df = df[df['horsepower'] != '?']
df['horsepower'] = df['horsepower'].astype(int)
print(df.isnull().sum())
print(df.nunique())

# Visualize average MPG by cylinders and origin
plt.subplots(figsize=(15, 5)) 
for i, col in enumerate(['cylinders', 'origin']): 
	plt.subplot(1, 2, i + 1) 
	df.groupby(col).mean()['mpg'].plot.bar() 
	plt.xticks(rotation=0) 
plt.tight_layout() 
plt.show() 

# Correlation heatmap
plt.figure(figsize=(8, 8)) 
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False) 
plt.show() 

# Drop less significant feature
df.drop('displacement', axis=1, inplace=True)

# Split data into features and target
features = df.drop(['mpg', 'car name'], axis=1) 
target = df['mpg'].values 
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=22) 
print(X_train.shape, X_val.shape)

# Create TensorFlow datasets
AUTO = tf.data.experimental.AUTOTUNE 
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(32).prefetch(AUTO)) 
val_ds = (tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(32).prefetch(AUTO)) 

# Build and compile model
model = keras.Sequential([ 
	layers.Dense(256, activation='relu', input_shape=[6]), 
	layers.BatchNormalization(), 
	layers.Dense(256, activation='relu'), 
	layers.Dropout(0.3), 
	layers.BatchNormalization(), 
	layers.Dense(1, activation='relu') 
]) 
model.compile(loss='mae', optimizer='adam', metrics=['mape']) 
model.summary()

# Train model
history = model.fit(train_ds, epochs=50, validation_data=val_ds)

# Plot training history
history_df = pd.DataFrame(history.history) 
history_df.loc[:, ['loss', 'val_loss']].plot() 
history_df.loc[:, ['mape', 'val_mape']].plot() 
plt.show()
