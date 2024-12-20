import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv('song_data1.csv')

feats = data[['dance', 'energy', 'loudness',"acousticness","instrumentalness","key"]].copy()


def scale(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    scaled = abs(x - mean) / std
    max = np.max(scaled, axis=0)
    min = np.min(scaled, axis=0)
    scaled = (scaled - min) / (max-min) * 10

    return scaled

x = data.drop(columns=['genre'])
x = x.apply(pd.to_numeric, errors='coerce')

x = x.fillna(x.mean())

x = scale(x)
print(x)
y = data['genre'].values

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=50)

def train_svm(X_train, y_train, X_test, y_test, kernel='linear'):
    print(f"\nTraining SVM with {kernel} kernel...")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 20, 40],
        'gamma': ['scale', 'auto'] if kernel in ['rbf', 'poly', 'sigmoid'] else None,
        'degree': [2, 3, 4] if kernel == 'poly' else None,
        'kernel': [kernel]
    }
    # Remove None values from param_grid
    param_grid = {k: v for k, v in param_grid.items() if v is not None}

    # Initialize SVM model
    svm = SVC()

    # Perform Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Best parameters
    print("Best parameters:", grid_search.best_params_)

    # random_search = RandomizedSearchCV(estimator=svm, param_distributions=param_grid, n_iter=50, cv=5)
    # random_search.fit(x_train, y_train)
    # print("Best parameters:", random_search.best_params_)

    # Train the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return best_model

# Train linear SVM
linear_svm = train_svm(x_train, y_train, x_test, y_test, kernel='linear')

# Train SVM with RBF kernel
rbf_svm = train_svm(x_train, y_train, x_test, y_test, kernel='rbf')

# Train SVM with polynomial kernel
poly_svm = train_svm(x_train, y_train, x_test, y_test, kernel='poly')

sigmoid_svm = train_svm(x_train, y_train, x_test, y_test, kernel='sigmoid')