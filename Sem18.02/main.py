import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import csv
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
import sklearn


def dataFromCsvNamed(fileName, isX):
    dataFromFile = []
    with open(fileName, 'rb') as f:
        reader = csv.reader(f)
        if isX == True:
            dataFromFile = [row for row in reader]
        else:
            for row in reader:
                dataFromFile = dataFromFile + row

    return dataFromFile

def writePredictionToFile(fileName, prediction):
    f = open(fileName,'w')
    f.write('Id,Solution\n')
    count = 1
    for x in prediction:
        f.write('{0},{1}\n'.format(count, x))
        count += 1

xTrain = np.array(dataFromCsvNamed('train.csv', True))
yTrain = np.array(dataFromCsvNamed('trainLabels.csv', False))
testSet = np.array(dataFromCsvNamed('test.csv', True))

#here we use brute forse to find best classifier
'''
def predictWithClassifier(classifier):
    #return accurasy score, slassifier
    classifier.fit(xTrain[:500], yTrain[:500])
    return (sklearn.metrics.accuracy_score(yTrain[500:], classifier.predict(xTrain[500:])), classifier)

classifiers = []
classifiers.append(predictWithClassifier(KNeighborsClassifier()))
classifiers.append(predictWithClassifier(SVC()))
classifiers.append(predictWithClassifier(DecisionTreeClassifier()))
classifiers.append(predictWithClassifier(AdaBoostClassifier()))

bestClassifier = max(classifiers)[1]
bestClassifier.fit(xTrain, yTrain)

writePredictionToFile('result.csv', bestClassifier.predict(testSet))
'''


#find optimum number of neighbors for kNN using grid;
#looking from 1 to 9 neigh
knn = KNeighborsClassifier()
parametrsDict = {'n_neighbors' : range(1, 10)}
clf = GridSearchCV(KNeighborsClassifier(), parametrsDict, cv=5, n_jobs=-1)
clf.fit(xTrain, yTrain)
print clf.predict(testSet)