import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import math
from scipy.constants._codata import val
data = pd.read_csv('song_data.csv')

feats = data[['dance', 'energy', 'loudness',"acousticness","instrumentalness","key"]].copy()



def one_hot_encode(labels, num_classes):
    one_hot_labels = np.zeros((labels.size, num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i, label - 1] = 1
    return one_hot_labels




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


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=50)
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(x_train, y_train)


y_pred = log_reg.predict(x_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
num_classes = len(np.unique(y))


y_train = one_hot_encode(y_train, num_classes)

weights = np.random.randn(x.shape[1], num_classes) * 1

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

    return weights, lossHistory


a,b = gradientDescent(x_train,y_train,weights,10000,0.001)
def predict(x, a):
    logits = np.dot(x, a)
    probabilities = softmax(logits)
    print(probabilities)
    return np.argmax(probabilities, axis=1) + 1  # Predicted class

def accuracy(predictions, y_test):
    false = 0
    for i in range(predictions.shape[0]):
        if predictions[i] == y_test[i]:
            false += 1
        else:
            false += 0
    return false / predictions.shape[0]
predictions = predict(x_test, weights)
print(accuracy(predictions, y_test))

