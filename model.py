import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

input_df = pd.read_csv("model_input_df_2.csv", usecols=["receiverx", "receivery", "receivers", "receivera", "receiverdis", "receivero", "receiverdir", "distance_to_nearest_def", "defenderx",
     "defendery", "defenders", "defendera", "defenderdis", "defendero", "defenderdir",
    "defenders_in_path","pass_length", "yards_to_go", "yardline_num", "yards_gained"])

#input_df = input_df[input_df["yards_gained"]<=22]
input_df = input_df[input_df["yards_gained"]>=0]
x = input_df.drop(columns=["yards_gained"])
y = input_df["yards_gained"]

counts, bins, _ = plt.hist(y, bins=76)
plt.xlabel('Yards Gained')
plt.ylabel('Frequency')
plt.title('Distribution of Yards Gained')

# Set x-axis ticks every 2 units
plt.xticks(np.arange(int(min(bins)), int(max(bins)) + 1, 2))

# Annotate the bars
for i in range(len(counts)):
    count = counts[i]
    bin_left = bins[i]
    bin_right = bins[i + 1]
    plt.text((bin_left + bin_right) / 2, count, str(int(count)), ha='center', va='bottom')

plt.show()


#80 percent for training 20 percent for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
#further split into validation sets 
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=7)

smote = SMOTE(sampling_strategy='auto', random_state=7)
x_train, y_train = smote.fit_resample(x_train, y_train)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_valid_scaled = scaler.transform(x_valid)

y_train = y_train.values.reshape(-1, 1)
y_valid = y_valid.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

#plt.hist(y_train, bins=23)
#plt.xlabel('Yards Gained')
#plt.ylabel('Frequency')
#plt.title('Distribution of Yards Gained')
#for i in range(len(plt.hist(y_train, bins=23)[0])):
#    count = plt.hist(y_train, bins=23)[0][i]
#    bin_left = plt.hist(y_train, bins=23)[1][i]
#    bin_right = plt.hist(y_train, bins=23)[1][i + 1]
#    plt.text((bin_left + bin_right) / 2, count, str(int(count)), ha='center', va='bottom')
#plt.show()

print(x_train_scaled.shape)

#model = tf.keras.Sequential([
#    tf.keras.layers.Dense(75, activation='relu', kernel_regularizer=regularizers.l2(0.00015)),
#    tf.keras.layers.BatchNormalization(),
#    tf.keras.layers.Dropout(0.3),
#    tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.00015)),
#    tf.keras.layers.BatchNormalization(),
#    tf.keras.layers.Dropout(0.3),
#    tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.00015)),
#    tf.keras.layers.Dense(1, activation='linear')
#])

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(75, activation='tanh', kernel_regularizer=regularizers.l2(0.00015), input_shape=(1,20)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(50, activation='tanh', kernel_regularizer=regularizers.l2(0.00015)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(25, activation='tanh', kernel_regularizer=regularizers.l2(0.00015)),
    tf.keras.layers.LSTM(1, activation='linear')
])

x_train_scaled = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))  # (samples, time_steps=1, features)
x_valid_scaled = x_valid_scaled.reshape((x_valid_scaled.shape[0], 1, x_valid_scaled.shape[1]))  # Same for validation
x_test_scaled = x_test_scaled.reshape((x_test_scaled.shape[0], 1, x_test_scaled.shape[1]))

print(x_train_scaled.shape)

model.compile(loss=tf.keras.losses.MeanSquaredError, optimizer = tf.keras.optimizers.Adam(learning_rate=0.0009), metrics=['mse'])
history = model.fit(
    x_train_scaled, 
    y_train, 
    epochs=100, 
    batch_size=20,
    validation_data=(x_valid_scaled,y_valid)
    )

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Time (Training and Validation)')
plt.legend()
plt.show()


y_pred = model.predict(x_test_scaled)
loss, mse = model.evaluate(x_test_scaled, y_test)
print(f"Test Loss: {loss}")
print(f"Test MSE: {mse}")

y_pred = np.round(y_pred)

y_pred = np.array(y_pred).flatten() 
y_test = np.array(y_test).flatten()

pairs = list(zip(y_test, y_pred))
unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
pair_size_map = dict(zip(map(tuple, unique_pairs), counts))
point_sizes = np.array([pair_size_map[tuple(val)] for val in pairs])

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', s=point_sizes*10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlim(0, y_test.max())
plt.ylim(0, y_test.max())
plt.xticks(np.arange(0, y_test.max()+1, 1))
plt.yticks(np.arange(0, y_test.max()+1, 1))
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()
