import numpy as np
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

class LinearSVM:
    def __init__(self, C=1.0):
        self.C = C  # Regularization parameter
        self.w = None
        self.b = None
        self.alpha = None

    def fit(self, X, y):
        y = y.reshape(-1, 1) * 1.0

        n_samples, n_features = X.shape

        # Compute the Gram matrix (dot products of all feature vectors)
        K = np.dot(X, X.T)

        # Convert to cvxopt format
        P = matrix(np.outer(y, y) * K)  # Quadratic term
        q = matrix(-np.ones((n_samples, 1)))  # Linear term
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))  # Inequality constraints
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))  # Bounds for alpha
        A = matrix(y.T, (1, n_samples))  # Equality constraint
        b = matrix(0.0)  # Equality constraint bias

        # Solve QP problem
        solvers.options['show_progress'] = False  # Suppress solver output
        solution = solvers.qp(P, q, G, h, A, b)

        # Extract Lagrange multipliers
        alpha = np.array(solution['x']).flatten()

        # Support vectors have non-zero Lagrange multipliers
        sv = alpha > 1e-5
        self.alpha = alpha[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]

        # Calculate weights
        self.w = np.sum(self.alpha[:, None] * self.sv_y * self.sv_X, axis=0)

        # Calculate bias
        self.b = np.mean(self.sv_y - np.dot(self.sv_X, self.w))

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

# One-vs-All Multi-Class Wrapper
class OneVsAllSVM:
    def __init__(self, C=1.0):
        self.C = C
        self.models = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            print(f"Training SVM for class {c} vs all with C={self.C}...")
            # Create binary labels for class c
            y_binary = np.where(y == c, 1, -1)
            model = LinearSVM(C=self.C)
            model.fit(X, y_binary)
            self.models[c] = model

    def predict(self, X):
        # Collect decision function values for each class
        decision_values = {c: model.decision_function(X) for c, model in self.models.items()}
        # Choose the class with the highest decision value
        decision_values = np.array([decision_values[c] for c in self.classes]).T
        return self.classes[np.argmax(decision_values, axis=1)]

# Load dataset
data = pd.read_csv('song_data1.csv')

# Features and target
x = data.drop(columns=['genre'])
x = x.apply(pd.to_numeric, errors='coerce')
x = x.fillna(x.mean())

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Target
y = data['genre'].values

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=50)

# Define range of C values to test
param_grid = {'C': [0.01, 0.1, 1, 10, 20]}

# Loop over different C values
for C in param_grid['C']:
    print(f"\nEvaluating SVM with C={C}")
    
    # Train One-vs-All SVM
    multi_svm = OneVsAllSVM(C=C)
    multi_svm.fit(x_train, y_train)

    # Make predictions
    y_pred = multi_svm.predict(x_test)

    # Evaluate model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
