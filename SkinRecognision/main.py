import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
import sklearn
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn import metrics


class PicturePredictor:

    #read information about points from file (fileName given)
    #return xTrain set with object's features and yTrain with kind of class
    #returning types is np.array
    #return data in train file if isTrain is True
    def dataFromCsvNamed(self, fileName, isTrain):
        dataFromFile = []
        with open(fileName, 'rb') as f:
            reader = csv.reader(f)
            reader.next()
            dataFromFile = [row for row in reader]

        dataFromFile = np.array(dataFromFile)

        if isTrain:
            return dataFromFile[:, range(3)].astype(int), dataFromFile[:, 3].astype(int)
        return dataFromFile[:, range(1, 4)].astype(int), None

    #write prediction to the file named fileName
    def writePredictionToFile(self, fileName, prediction):
        f = open(fileName,'w')
        f.write('id,y\n')
        count = 1
        for x in prediction:
            f.write('{0},{1}\n'.format(count, x))
            count += 1

    #return kNN clasifier with fitted params
    def trainClassifierWithData(self, xTrain, yTrain):
        clf = KNeighborsClassifier()
        parametrsDict = {'n_neighbors' : range(1, 10), 'metric' : ['manhattan', 'chebyshev']}
        clf = GridSearchCV(KNeighborsClassifier(), parametrsDict, cv=5, n_jobs=-1)
        clf.fit(xTrain, yTrain)
        return clf

    def __init__(self):
        self.xTrain, self.yTrain = self.dataFromCsvNamed('train.csv', True)
        #self.knn = self.trainClassifierWithData(self.xTrain, self.yTrain)

    def loadTestSetFromFile(self):
        self.xTest, self.yTest = self.dataFromCsvNamed('test.csv', False)

    def loadTestSetFromArray(self, set):
        self.xTest = set

    def predictData(self):
        return self.knn.predict(self.xTest)

    def createRandomForest(self, numberOfTrees):
        self.forest = []

        m = len(self.xTrain) / 2

        X_train, self.X_test = self.xTrain[:m], self.xTrain[m:]
        Y_train, self.Y_test = self.yTrain[:m], self.yTrain[m:]

        for i in xrange(numberOfTrees):
            tree = DecisionTreeClassifier(max_features=2)
            tree.fit(X_train, Y_train)
            self.forest.append(tree)

    def predict(self):
        predictions = []
        for tree in self.forest:
            yPred= tree.predict(self.X_test)
            predictions.append(yPred)
        

pic = PicturePredictor()
pic.createRandomForest(5)
pic.predict()