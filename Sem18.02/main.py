import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import csv

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
        f.write('%d,%d\n' % (count,x))
        count += 1

xTrain = np.array(dataFromCsvNamed('train.csv', True))
yTrain = np.array(dataFromCsvNamed('trainLabels.csv', False))
testSet = np.array(dataFromCsvNamed('test.csv', True))

knn = KNeighborsClassifier()
knn.fit(xTrain, yTrain)

prediction = knn.predict(testSet)

writePredictionToFile('result.csv', prediction.astype(int))

'''
myfile = open('result.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(prediction.astype(np.int))
'''
#out = csv.writer(open("result.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
#out.writerows(prediction)

#np.savetxt("result.csv", prediction.astype(np.float), delimiter = ",")

#print knn.predict(testSet)