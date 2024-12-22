## Random Forest Classifier Using Scikit-Learn


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import time

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
    'n_estimators': [10, 25, 50, 100, 200], 
    'max_features': ['sqrt', 'log2', None], 
    'max_depth': [3, 6, 9, None], 
    'max_leaf_nodes': [3, 6, 9, None], 
} 

#Train Random Forest
model = RandomForestClassifier() 
start_time = time.time()
model.fit(x_train, y_train) 
end_time = time.time()
y_pred = model.predict(x_test)                   # without parameter optimization

print(f"Training + testing runtime: {end_time - start_time:.2f}s")
  
# performance evaluation metrics 
print(classification_report(y_pred, y_test))  # avg % 87 precision 





grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)        # grid search with 5 fold cross validation
start_time = time.time()
grid_search.fit(x_train, y_train) 
end_time = time.time()
print(grid_search.best_estimator_)                                                       # best estimators
print(f"Time taken for grid search: {end_time - start_time:.2f}s")                       # time taken for grid search


model_grid = RandomForestClassifier(max_depth=grid_search.best_params_['max_depth'],
        max_features=grid_search.best_params_['max_features'],
        max_leaf_nodes=grid_search.best_params_['max_leaf_nodes'],
        n_estimators=grid_search.best_params_['n_estimators']) 

start_time = time.time()
model_grid.fit(x_train, y_train) 
y_pred_grid = model_grid.predict(x_test) 
end_time = time.time()

print(classification_report(y_pred_grid, y_test))                                    # model performance with grid search
print(f"Training + testing runtime: {end_time - start_time:.2f} seconds")                # training runtime




random_search = RandomizedSearchCV(RandomForestClassifier(), param_grid, cv=5)            # random search with 5 fold cross validation
start_time = time.time()
random_search.fit(x_train, y_train) 
end_time = time.time()
print(random_search.best_estimator_)                                                   # best estimators
print(f"Time taken for random search: {end_time - start_time:.2f}s")                                      # time taken for random search

model_grid = RandomForestClassifier(max_depth=random_search.best_params_['max_depth'],
        max_features=random_search.best_params_['max_features'],
        max_leaf_nodes=random_search.best_params_['max_leaf_nodes'],
        n_estimators=random_search.best_params_['n_estimators']) 

start_time = time.time()
model_grid.fit(x_train, y_train) 
y_pred_grid = model_grid.predict(x_test) 
end_time = time.time()

print(classification_report(y_pred_grid, y_test))                                    # model performance with random search
print(f"Training + testing runtime: {end_time - start_time:.2f} seconds")                      # training runtime