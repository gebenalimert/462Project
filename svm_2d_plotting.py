## Linear Soft SVM Implementation with cvxopt library Plotted in 2D


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from cvxopt import matrix, solvers          # library for dual optimization
import matplotlib.pyplot as plt

# Linear Soft-Margin SVM
class LinearSoftMarginSVM:
    def __init__(self, C=1.0):
        self.C = C  # Regularization parameter
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        y = y.reshape(-1, 1) * 1.0  # class values are -1 and 1, handled in the data preprocessing

        n_samples, n_features = X.shape

        # Compute the Gram matrix
        K = np.dot(X, X.T)

        # Convert parameters to cvxopt format
        Q = matrix(np.outer(y, y) * K)
        p = matrix(-np.ones((n_samples, 1)))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = matrix(y.T, (1, n_samples))
        b = matrix(0.0)

        # Solve QP problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(Q, p, G, h, A, b)

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
        return np.sign(self.decision_function(X))    # determining classes based on the sign of the decision function

# Load dataset
data = pd.read_csv('song_data1.csv')

# Filter the dataset to include only classes 2 and 3
data = data[data['genre'].isin([2, 3])]                # we picked classes 2 and 3 based on the accuracy results, you can change it

# Features and target
x = data.drop(columns=['genre'])
x = x.apply(pd.to_numeric, errors='coerce')
x = x.fillna(x.mean())

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply PCA to reduce feature dimensions to 2
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

# Target
y = data['genre'].values
y = np.where(y == 2, -1, 1)  # Convert labels to -1 and 1 for SVM       !!! If you will use different classes, you should change this line

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=50)

# Train SVM on 2D data
svm_2d = LinearSoftMarginSVM(C=1.0)
svm_2d.fit(x_train, y_train)

# Predict on the 2D test set
y_pred_2d = svm_2d.predict(x_test)

# Evaluate the model on the 2D test set
print("Classification Report for 2D Test Set:")
print(classification_report(y_test, y_pred_2d))
print(f"Accuracy on 2D Test Set: {accuracy_score(y_test, y_pred_2d) * 100:.2f}%")

# Create a mesh grid for decision boundary visualization
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Compute decision function values for the mesh grid
Z = svm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the training points
plt.figure(figsize=(10, 6))
for label, marker, color in zip([-1, 1], ['o', 'x'], ['red', 'blue']):
    plt.scatter(x_train[y_train == label][:, 0],
                x_train[y_train == label][:, 1],
                marker=marker, color=color, label=f"Class {label}")

# Plot the decision boundary
plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=['pink', 'lightblue'], alpha=0.5)
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=1.5)

# Add labels and legend
plt.title("Linear SVM Decision Boundary with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
