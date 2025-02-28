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


df = pd.read_csv('laptop_price_dataset.csv')


encoder = LabelEncoder()

df['Company'] = encoder.fit_transform(df['Company'])
df['Product'] = encoder.fit_transform(df['Product'])
df['TypeName'] = encoder.fit_transform(df['TypeName'])
df['ScreenResolution'] = encoder.fit_transform(df['ScreenResolution'])
df['CPU_Company'] = encoder.fit_transform(df['CPU_Company'])
df['CPU_Type'] = encoder.fit_transform(df['CPU_Type'])
df['Memory'] = encoder.fit_transform(df['Memory'])
df['GPU_Company'] = encoder.fit_transform(df['GPU_Company'])
df['GPU_Type'] = encoder.fit_transform(df['GPU_Type'])
df['OpSys'] = encoder.fit_transform(df['OpSys'])

df.head()


x = df.iloc[ : , 0:-1]
y = df.iloc[ : , -1]

scaler = MinMaxScaler()

x = scaler.fit_transform(x)

# Convert to DataFrame
y_df = y.to_frame()

# Fit and transform y
y = scaler.fit_transform(y_df)

#Train & Test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

model_1 = LinearRegression()
model_2 = SGDRegressor()
model_3 = Lasso()
model_4 = Ridge()
model_5 = ElasticNet()
model_6 = SVR()
model_7 = KNeighborsRegressor()
model_8 = DecisionTreeRegressor()
model_9 = RandomForestRegressor()
model_10 = BaggingRegressor()
model_11 = ExtraTreesRegressor()
model_12 = AdaBoostRegressor()
model_13 = XGBRegressor(verbose=0)
model_14 = CatBoostRegressor(verbose=0)
model_15 = LGBMRegressor(verbose=0)

model_1.fit(x_train,y_train)
model_2.fit(x_train,y_train)
model_3.fit(x_train,y_train)
model_4.fit(x_train,y_train)
model_5.fit(x_train,y_train)
model_6.fit(x_train,y_train)
model_7.fit(x_train,y_train)
model_8.fit(x_train,y_train)
model_9.fit(x_train,y_train)
model_10.fit(x_train,y_train)
model_11.fit(x_train,y_train)
model_12.fit(x_train,y_train)
model_13.fit(x_train,y_train)
model_14.fit(x_train,y_train)
model_15.fit(x_train,y_train)

df.columns

models = [model_1, model_2, model_3, model_4, model_5,
          model_6, model_7, model_8, model_9, model_10,
          model_11, model_12, model_13, model_14, model_15]
models_names = ['LinearRegression', 'SGDRegressor', 'Lasso', 'Ridge', 'ElasticNet', 'SVR', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor', 'BaggingRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor', 'XGBRegressor', 'CatBoostRegressor', 'LGBMRegressor']

# Calculate predictions and squared errors for each model:
squared_errors = []
for model in models:
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    squared_errors.append(f'{mse * 100:.2f}%')  # Format as percentage

# Calculate train and test scores:
train_score = [model.score(x_train, y_train) for model in models]
test_score = [model.score(x_test, y_test) for model in models]

# Difference between training and testing ratio
ratio = []
for train, test in zip(train_score, test_score):
    result = train - test
    ratio.append(f'{result * 100:.2f}%')

# Measure model state:6
rate = []
for train, test in zip(train_score, test_score):
    if train <= 0.65 and test <= 0.65:
        rate.append('bad')
    elif train > test * 1.10:
        rate.append('overfite')
    elif train > 0.65 and train < 0.80 and test > 0.65 and test < 0.80:
        rate.append('middle')
    elif train >= 0.80 and test >= 0.80 and train < 1.00 and test < 1.00:
        rate.append('good')
    elif train >= 0.80 and test < 0.80:
        rate.append('high train, low test')
    else:
        rate.append('unknown')

# Create DataFrame
model_score = pd.DataFrame({
    'Model': models_names,
    'Train score': [f'{round(score * 100, 2)}%' for score in train_score],
    'Test score': [f'{round(score * 100, 2)}%' for score in test_score],
    'Ratio difference': ratio,
    'Evaluate model': rate,
    'Squared error': squared_errors
})

# Show result:
model_score

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Initialize the XGBRegressor
xgb_model = XGBRegressor(verbose=0)

# Initialize the GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# Fit the model
grid_search.fit(x_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Use the best estimator to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

# Evaluate the model

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


