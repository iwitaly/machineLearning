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
        #list with DecisionTreeClassifier's
        self.forest = []

        m = len(self.xTrain) / 2

        #self second half of set as a class parametr
        X_train, self.X_test = self.xTrain[:m], self.xTrain[m:]
        Y_train, self.Y_test = self.yTrain[:m], self.yTrain[m:]

        #create DecisionTreeClassifier's and fit them
        for i in xrange(numberOfTrees):
            tree = DecisionTreeClassifier(max_features=2)
            tree.fit(X_train, Y_train)
            self.forest.append(tree)

    def predict(self):
        #1) predict with classifiers from self.forest
        predictions = []
        for tree in self.forest:
            yPred = tree.predict(self.X_test)
            predictions.append(yPred)

        #2) create summery preduction as a max votes from every tree
        summeryPrediction = []
        for i in range(len(predictions[0])):
            numberOfPluses = 0
            numberOfMinuses = 0
            for predic in predictions:
                if predic[i] == 1:
                    numberOfPluses += 1
                else:
                    numberOfMinuses += 1
            if numberOfPluses > numberOfMinuses:
                summeryPrediction.append(1)
            else:
                summeryPrediction.append(-1)

        #do something else
        print metrics.accuracy_score(self.Y_test, summeryPrediction)


pic = PicturePredictor()
pic.createRandomForest(10)
pic.predict()