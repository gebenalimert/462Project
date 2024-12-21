## Linear Soft SVM Implementation with cvxopt library


import numpy as np
from cvxopt import matrix, solvers                       # library for dual optimization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Linear Soft-Margin SVM
class LinearSoftMarginSVM:
    def __init__(self, C=1.0):
        self.C = C  # Regularization parameter
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        y = y.reshape(-1, 1) * 1.0  # Class values should be -1 and 1

        n_samples, n_features = X.shape

        # Compute the Gram matrix
        K = np.dot(X, X.T)

        # Convert parameters to cvxopt format
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones((n_samples, 1)))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = matrix(y.T, (1, n_samples))
        b = matrix(0.0)

        # Solve QP problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)

        # Extract Lagrange multipliers
        alphas = np.array(solution['x']).flatten()

        # Support vectors have non-zero Lagrange multipliers
        support_vector_indices = alphas > 1e-5
        self.alphas = alphas[support_vector_indices]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]

        # Compute weight vector
        self.w = np.sum(self.alphas[:, None] * self.support_vector_labels * self.support_vectors, axis=0)

        # Compute bias term
        self.b = np.mean(
            self.support_vector_labels - np.dot(self.support_vectors, self.w)
        )

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))           # determining classes based on the sign of the decision function

# Load dataset
data = pd.read_csv('song_data1.csv')

# Filter the dataset to include only classes 1 and 2, this can be changed to include other classes
data = data[data['genre'].isin([1, 2])]

# Features and target
x = data.drop(columns=['genre'])
x = x.apply(pd.to_numeric, errors='coerce')
x = x.fillna(x.mean())

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Target
y = data['genre'].values

# Convert labels to -1 and 1 for SVM
y = np.where(y == 1, -1, 1)              #   !!! if you are using different classes, you need to change this line

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=50)

# Train the SVM
svm = LinearSoftMarginSVM(C=1.0)
svm.fit(x_train, y_train)

# Make predictions
y_pred = svm.predict(x_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
