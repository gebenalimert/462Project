import numpy as np
import pandas as pd

data = pd.read_csv('song_data.csv')

feats = data[['dance', 'energy', 'loudness',"acousticness","instrumentalness","key"]].copy()
import time
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
x = np.array(x)

y = data['genre'].values

def kFold(x,y,k):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    folds = np.array_split(indices,5)
    results = {}
    runTimes = {}
    for number in k:
        results[number] = []
        runTimes[number] = []
    for i in range(5):
        testIndices = folds[i]
        trainList = []

        for j in range(5):
            if j != i:
                trainList.append(folds[j])

        trainIndices = np.concatenate(trainList)
        xTrain, yTrain = x[trainIndices], y[trainIndices]
        xTest, yTest = x[testIndices], y[testIndices]

    for values in k:
            startTime =time.time()
            predictions = predict(xTrain,xTest, yTrain, values)
            endTime =time.time()
            # Calculate accuracy
            true = 0
            for i in range(len(predictions)):
                if predictions[i] == yTest[i]:
                    true += 1
            acc = true/len(predictions)
            results[values].append(acc)
            runtime = endTime - startTime
            runTimes[values].append(runtime)

        # Compute average accuracy for each k
    avgResults = {}
    avgRuntimes = {}
    for k, accuracy in results.items():
        average = np.mean(accuracy)
        avgResults[k] = average
    for k, run in results.items():
        average = np.mean(run)
        avgRuntimes[k] = average
    # Find the best k
    bestk = max(avgResults, key=avgResults.get)

    return bestk, avgResults,avgRuntimes



def mahDistance(data, point):
    data = np.array(data)
    point = np.array(point)
    dataCov = np.cov(data, rowvar=0)
    dataCovInv = np.linalg.inv(dataCov)
    diff = data - point  # Calculate difference for all rows

    left = np.dot(diff, dataCovInv)  # Matrix multiplication
    dists = np.sqrt(np.sum(left * diff, axis=1))

    return dists

def predict(xTrain,xTest,y,k):
    predictions = []


    for point in xTest:
        distances = mahDistance(xTrain,point)
        kNearest = np.argsort(distances)[:k]
        nearest_labels = y[kNearest]
        predicted_label = np.bincount(nearest_labels).argmax()
        predictions.append(predicted_label)

    return np.array(predictions)
bestK, averageAcc,averageRun = kFold(x,y,[2,3,4,5,6,7,8,9,10])
print("Determined K via cross-validation: " + str(bestK))
print("Average Accuracy: " + str(averageAcc[bestK]))
print("Average Runtime: " + str(averageRun[bestK] * 1000) + " ms")

