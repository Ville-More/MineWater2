import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


#Load data

GB_model = pd.read_csv('ML_Model_Final.csv')

print(GB_model)

#Identify dependent (y) and independent variables (X).
#Dependent variable will be EC and pH while the independent variables will be Turbidity, TDS, SO4 and Fe

X = GB_model[['Turbidity', 'TotalDissolvedSolids', 'Sulfate', 'Iron']]
y = GB_model[['Electrical_Conductivity', 'pH']]

print(X)
print(y)

# Splitting the dataset into the Training set and Test set
# Data set will be split using test_size=0.2, meaning 20% of data rows will only be used as test set and the remaining rows will be used as training set for building the Gradient Boosting Regression Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

params = {'n_estimators': 100, 'max_depth': 2, 'min_samples_split': 2, 'min_samples_leaf': 2, 'learning_rate': 0.05
         }

# Model training and evaluation
# Train Gradient Boosting Regressor
# Important parameters = max_depth, n_estimators and learning_rate

gbModel = MultiOutputRegressor(GradientBoostingRegressor(**params))

# Train gradient boosting regressor
# Fit the model for the training data

gbModel.fit(X_train, y_train)

#Saving the model to disk

pickle.dump(gbModel, open('Final_MultiOutputGB_Model.pkl', 'wb'))

