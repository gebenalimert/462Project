## Linear Soft SVM and Logistic Regression Decision Boundaries Plotted Together

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from cvxopt import matrix, solvers
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
        return np.sign(self.decision_function(X))  # determining classes based on the sign of the decision function

# Logistic Regression Implementation
def one_hot_encode(labels, num_classes):
    one_hot_labels = np.zeros((labels.size, num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i, label - 1] = 1
    return one_hot_labels

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability trick
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # Normalize across classes

def crossEntropyLoss(x,y,weights):
    y_hat = softmax(np.dot(x,weights))
    loss = -np.sum(y * np.log(y_hat + 1e-15)) / (y.shape[0])  # Avoid log(0)
    #output = np.multiply(y, np.dot(x, weights.T))  # y_n * (w^T * x_n) ???Y_N OR
    #loss = np.log(1 + np.exp(-output)).mean()
    return loss

def gradient(weights, x, y):
    #output = y * np.dot(x, w)

    y_hat = softmax(np.dot(x, weights))
    output = y_hat - y
    grad = np.dot(x.T,output) / x.shape[0]
    #grad = -np.dot(x.T, y * (1 / (1 + np.exp(output)))) / len(x)
    return grad


def gradientDescent(x,y,weights,maxIter,learningRate):
    lossHistory = []
    for i in range(maxIter):

        # Compute loss
        loss = crossEntropyLoss(x,y,weights)
        lossHistory.append(loss)

        # Compute gradient
        grad = gradient(weights,x,y)
        # Update weights
        weights -= learningRate * grad
        if i % 100 == 0:  # Print loss every 100 iterations
            print(f"Iteration {i}: Loss = {loss:.4f}")
            if (i != 0) and (np.equal(round(lossHistory[i],6),round(lossHistory[i-5],6))):
                break
    return weights, lossHistory

def predict(x, a):
    logits = np.dot(x, a)
    probabilities = softmax(logits)
    return np.argmax(probabilities, axis=1) + 1  # Predicted class

def accuracy(predictions, y_test):
    false = 0
    for i in range(predictions.shape[0]):
        if predictions[i] == y_test[i]:
            false += 1
        else:
            false += 0
    return false / predictions.shape[0]



# Load dataset
data = pd.read_csv('song_data.csv')

# Filter the dataset to include only classes 2 and 3
data = data[data['genre'].isin([1, 2])]

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
y = np.where(y == 1, -1, 1)          # Convert labels to -1 and 1 for SVM     !!! If you will you different classes, you need to change this line

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=50)

# Train SVM on 2D data
svm_2d = LinearSoftMarginSVM(C=1.0)
svm_2d.fit(x_train, y_train)

# Predict on the 2D test set
y_pred_2d = svm_2d.predict(x_test)

# Train Logistic Regression on 2D data
num_classes = len(np.unique(y))

y_train_original = y_train  # To keep labels (-1 or 1)
y_train = one_hot_encode(y_train, num_classes)
weights = np.random.randn(x_train.shape[1], num_classes) * 1

a,b = gradientDescent(x_train,y_train,weights,100000,0.001)
predictions = predict(x_test, weights)
print(accuracy(predictions, y_test))

train_predictions = predict(x_train, a)
train_accuracy = accuracy(train_predictions, np.argmax(y_train, axis=1) + 1)
print("Training Accuracy:", train_accuracy)

test_predictions = predict(x_test, a)
test_accuracy = accuracy(test_predictions, y_test)
print("Test Accuracy:", test_accuracy)

train_loss = crossEntropyLoss(x_train, y_train, a)
test_loss = crossEntropyLoss(x_test, one_hot_encode(y_test, num_classes), a)
print("Training Loss:", train_loss)
print("Test Loss:", test_loss)


# Create a mesh grid for decision boundary visualization
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Compute decision function values for the mesh grid (SVM)
Z_svm = svm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)


# Compute decision function values for the mesh grid (Logistic Regression)
logits = np.dot(np.c_[xx.ravel(), yy.ravel()], a) # Adjust for the 2D PCA input
Z_log = np.max(softmax(logits), axis=1)  # Use softmax to calculate probabilities
Z_log = np.argmax(logits, axis=1).reshape(xx.shape)  # Decision boundaries based on the class probabilities


loading_matrix = pca.components_
feature_names = x.columns

top_features_pca1 = np.argsort(np.abs(loading_matrix[0]))[-2:]  # Indices of top 2 features for PCA1
top_features_pca2 = np.argsort(np.abs(loading_matrix[1]))[-2:]  # Indices of top 2 features for PCA2

pca1_main_features = ", ".join([feature_names[i] for i in top_features_pca1])
pca2_main_features = ", ".join([feature_names[i] for i in top_features_pca2])




# Plot the training points
plt.figure(figsize=(12, 8))
for label, marker, color in zip([-1, 1], ['o', 'x'], ['red', 'blue']):
    mask = (y_train_original == label)  # Use the original 1D class labels for masking
    plt.scatter(x_train[mask][:, 0], x_train[mask][:, 1], marker=marker, color=color, label=f"Class {label}")


# Plot the decision boundary (SVM)
plt.contour(xx, yy, Z_svm, levels=[0], colors='green', linewidths=1.5, linestyles='dashed', label="SVM Boundary")

# Plot the decision boundary (Logistic Regression)
plt.contour(xx, yy, Z_log, levels=[0.5], colors='orange', linewidths=1.5, linestyles='solid', label="Logistic Regression Boundary")

# Add labels and legend
plt.title("Decision Boundaries: Linear SVM vs Logistic Regression")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(loc='upper right', title='Legend')
plt.annotate("Dashed: SVM Boundary\nSolid: Logistic Regression", xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10, color='black')
# Annotate PCA contributions
plt.annotate(f"PCA1: {pca1_main_features}", xy=(0.02, 0.9), xycoords='axes fraction', fontsize=10, color='black')
plt.annotate(f"PCA2: {pca2_main_features}", xy=(0.02, 0.85), xycoords='axes fraction', fontsize=10, color='black')

plt.show()

