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

input_df = pd.read_csv("model_input_df_2.csv", usecols=["receiverx", "receivery", "receivers", "receivera", "receiverdis", "receivero", "receiverdir", "distance_to_nearest_def", 
    "defenders_in_path","pass_length", "yards_to_go", "yardline_num", "yards_gained"])

input_df = input_df[input_df["yards_gained"]<=15]
input_df = input_df[input_df["yards_gained"]>=0]
x = input_df.drop(columns=["yards_gained"])
y = input_df["yards_gained"]

#80 percent for training 20 percent for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
#further split into validation sets 
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=7)

#smote = SMOTE(sampling_strategy='auto', random_state=7)
#x_train, y_train = smote.fit_resample(x_train, y_train)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_valid_scaled = y_scaler.transform(y_valid.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

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



#feedfoward 100% our work
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create the model
#model = Sequential()

# Add first hidden layer with 50 units and ReLU activation
#model.add(Dense(50, activation='relu', input_shape=(input_dim,)))  # input_dim should be your number of features

# Add second hidden layer with 50 units and ReLU activation
#model.add(Dense(50, activation='relu'))

# Add output layer (adjust units and activation based on your task)
# For binary classification:
# model.add(Dense(1, activation='sigmoid'))
# For multi-class classification:
# model.add(Dense(num_classes, activation='softmax'))
# For regression:
# model.add(Dense(1))  # linear activation

# Compile the model (adjust based on your task)
# For classification:
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# For regression:
# model.compile(optimizer='adam', loss='mse')

# Print model summary



#regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Example with 10 input features
#model = Sequential([
#    Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0002)),
#   tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.Dropout(0.2),
#    Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0002)),
#    tf.keras.layers.Dropout(0.2),
#    Dense(1) 
#])

#model.compile(optimizer='adam', loss='mse', metrics=['mse'])
#history = model.fit(
#    x_train_scaled, 
#    y_train, 
#    epochs=100, 
#    batch_size=128,
#    validation_data=(x_valid_scaled,y_valid)
#    )
#model.summary()


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

# Define the number of folds
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=7)

# Store results
results = []

# Define parameter combinations to test
param_combinations = [
    {'units': 256, 'dropout': 0.1, 'l1_reg': 0.002}
]
counter = 0
for params in param_combinations:
    fold_scores = []

    for train_idx, val_idx in kf.split(x_train_scaled):
        # Split data
        X_train_fold, X_val_fold = x_train_scaled[train_idx], x_train_scaled[val_idx]
        y_train_fold, y_val_fold = y_train_scaled[train_idx], y_train_scaled[val_idx]
    
        
        # Build model
        model = Sequential([
            Dense((params['units']), activation='relu', 
                  kernel_regularizer=regularizers.l1(params['l1_reg'])),
            tf.keras.layers.Dropout(params['dropout']),
            Dense((params['units']), activation='relu',
                  kernel_regularizer=regularizers.l1(params['l1_reg'])),
            tf.keras.layers.Dropout(params['dropout']),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        
        # Train with early stopping
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=100,
            batch_size=64,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        )
        
        # Store best validation score
        best_val_loss = min(history.history['val_loss'])
        fold_scores.append(best_val_loss)
    
    # Store average performance across folds
    results.append({
        'params': params,
        'mean_val_loss': np.mean(fold_scores),
        'std_val_loss': np.std(fold_scores)
    })

# Find best parameters
best_result = min(results, key=lambda x: x['mean_val_loss'])
print(f"Best parameters: {best_result['params']}")
print(f"Best validation loss: {best_result['mean_val_loss']:.4f} ± {best_result['std_val_loss']:.4f}")

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
loss, mse = model.evaluate(x_test_scaled, y_test_scaled)
print(f"Test Loss: {loss}")
print(f"Test MSE: {mse}")
y_pred = y_scaler.inverse_transform(y_pred)
y_test = y_scaler.inverse_transform(y_test_scaled)
mse_unscaled = mean_squared_error(y_test, y_pred)
print(f"Unscaled MSE (yards): {mse_unscaled:.2f}")



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

from sklearn.metrics import r2_score, mean_absolute_error

y_pred = model.predict(x_test_scaled)
y_pred_original = y_scaler.inverse_transform(y_pred)
y_test_original = y_scaler.inverse_transform(y_test_scaled)

print(f"MAE: {mean_absolute_error(y_test_original, y_pred_original)}")
print(f"R²: {r2_score(y_test_original, y_pred_original)}")

residuals = y_test_original - y_pred_original
plt.hist(residuals, bins=20)
plt.title("Distribution of Residuals")
plt.show()