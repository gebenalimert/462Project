## Random Forest Classifier Using Scikit-Learn


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('song_data1.csv')   # reading data from csv file

# Features and target
x = data.drop(columns=['genre'])
x = x.apply(pd.to_numeric, errors='coerce')
x = x.fillna(x.mean())                 # filling missing values with mean
y = data['genre'].values

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=50)   # splitting data into training and testing data

param_grid = {                                  # defining parameters for grid and random search
    'n_estimators': [25, 50, 100, 150], 
    'max_features': ['sqrt', 'log2', None], 
    'max_depth': [3, 6, 9], 
    'max_leaf_nodes': [3, 6, 9], 
} 

# Train Random Forest
model = RandomForestClassifier() 
model.fit(x_train, y_train) 

# predict the mode 
# y_pred = model.predict(x_test)                   # without parameter optimization
  
# performance evaluation metrics 
# print(classification_report(y_pred, y_test))  # avg % 87 precision 

# grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)        # grid search with 5 fold cross validation
# grid_search.fit(x_train, y_train) 
# print(grid_search.best_estimator_)                                                       # best estimators


# model_grid = RandomForestClassifier(max_depth=9, 
#                                     max_features=None, 
#                                     max_leaf_nodes=9, 
#                                     n_estimators=50) 
# model_grid.fit(x_train, y_train) 
# y_pred_grid = model.predict(x_test) 
# print(classification_report(y_pred_grid, y_test)) 


random_search = RandomizedSearchCV(RandomForestClassifier(), param_grid, cv=5)            # random search with 5 fold cross validation
random_search.fit(x_train, y_train) 
print(random_search.best_estimator_) 

model_grid = RandomForestClassifier(max_depth=9, 
                                    max_features='log2', 
                                    max_leaf_nodes=9, 
                                    n_estimators=50) 
model_grid.fit(x_train, y_train) 
y_pred_grid = model.predict(x_test) 
print(classification_report(y_pred_grid, y_test)) 