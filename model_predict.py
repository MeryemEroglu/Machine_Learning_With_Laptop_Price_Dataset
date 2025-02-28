import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from model_preprocessing import *
# Best parameters from grid search (best_params) can be used for feature engineering
# Assuming best_model is already fitted
best_model = grid_search.best_estimator_

# Make predictions with the best model
best_predictions = best_model.predict(x_test)

# Create a neural network model using Keras/TensorFlow
nn_model = Sequential()

# Input layer (number of features from the dataset)
nn_model.add(Dense(units=64, input_dim=x_train.shape[1], activation='relu'))

# Hidden layers (you can add more layers if needed)
nn_model.add(Dense(units=64, activation='relu'))
nn_model.add(Dropout(0.3))  # Dropout layer for regularization

# Output layer (single value output for regression)
nn_model.add(Dense(units=1))

# Compile the model
nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = nn_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
y_nn_pred = nn_model.predict(x_test)

# Calculate mean squared error of the predictions
mse_nn = mean_squared_error(y_test, y_nn_pred)

# Print out the results
print(f"Mean Squared Error of Neural Network: {mse_nn:.4f}")

# You can also visualize the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

