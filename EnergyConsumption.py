import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Read Datasets
train =  pd.read_csv('train_energy_data.csv')
test  =  pd.read_csv('test_energy_data.csv')

# Data preprocessing
train = pd.get_dummies(train, columns=["Building Type", "Day of Week"], drop_first=True)
test = pd.get_dummies(test, columns=["Building Type", "Day of Week"], drop_first=True)

# train and test data splitting
x_train = train.drop("Energy Consumption", axis=1)
y_train = train["Energy Consumption"]

x_test = test.drop("Energy Consumption", axis=1)
y_test = test["Energy Consumption"]

# Algorithm Selection
EnergyModel = LinearRegression()

#Traing the Model
EnergyModel.fit(x_train, y_train)

# Predicting the values

y_pred = EnergyModel.predict(x_test)

# Testing model efficiency
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Result is as under
#   MAE: 0.01216190115424979
#   RMSE: 0.014196563371695354
#   R2 Score: 0.9999999997063025

# Creating dump of trained model
joblib.dump(EnergyModel, 'Energy_Consumption_Model.joblib')
joblib.dump(x_train.columns.tolist(), "feature_names.joblib")