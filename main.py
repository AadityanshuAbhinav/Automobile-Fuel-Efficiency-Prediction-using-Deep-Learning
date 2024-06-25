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
df['horsepower'].unique()

# Clean data
df = df[df['horsepower'] != '?']
# Convert horsepower to numeric, handling errors
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce') 
# Fill missing values (NaN) in horsepower with 0
df['horsepower'] = df['horsepower'].fillna(0)
df['horsepower'] = df['horsepower'].astype(int) # Now convert to int
print(df.isnull().sum())
print(df.nunique())

# Visualize average MPG by cylinders and origin
plt.subplots(figsize=(15, 5)) 
for i, col in enumerate(['cylinders', 'origin']): 
	plt.subplot(1, 2, i + 1) 
	# Check if the column exists before grouping
	if col in df.columns:
		df.groupby(col).mean()['mpg'].plot.bar() 
		plt.xticks(rotation=0) 
	else:
		print(f"Column '{col}' not found in DataFrame.")
plt.tight_layout() 
plt.show() 

# # Correlation heatmap
# plt.figure(figsize=(8, 8)) 
# sb.heatmap(df.corr() > 0.9, annot=True, cbar=False) 
# plt.show() 

# Drop less significant feature
df.drop('displacement', axis=1, inplace=True)

from sklearn.model_selection import train_test_split
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
	layers.Dense(256, activation='relu', input_shape=[features.shape[1]]),
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
history_df.head()
history_df.loc[:, ['loss', 'val_loss']].plot() 
history_df.loc[:, ['mape', 'val_mape']].plot() 
plt.show()

# Example new data
new_data = pd.DataFrame({
    'cylinders': [4, 6, 8],
    'displacement': [140.0, 250.0, 350.0],
    'horsepower': [90, 180, 210],
    'weight': [2800, 3200, 3600],
    'acceleration': [15.0, 12.0, 11.0],
    'model-year': [82, 76, 79],
})

print("New data: ", new_data)

# Ensure the new data has the same preprocessing as the training data
new_data['horsepower'] = new_data['horsepower'].replace('?', np.nan).astype(float).fillna(0).astype(int)

# Convert the DataFrame to a NumPy array
new_data_array = new_data.to_numpy()

# Drop the 'model-year' column as it might not be used in training
new_data_array = new_data.drop('displacement', axis=1).to_numpy()

# Make predictions
predictions = model.predict(new_data_array)

# Display the predictions
print("Predicted mpg values:", predictions.flatten())
