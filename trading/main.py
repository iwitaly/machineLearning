__author__ = 'iwitaly'
import pandas
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

#<DATE>  <TIME>   <OPEN>   <HIGH>    <LOW>  <CLOSE>  <VOL>

#return X and Y
def readFile(fileName):
    f = pandas.read_csv(fileName)
    Y = np.array(f['<CLOSE>'])

    return f.as_matrix(['<OPEN>', '<HIGH>', '<LOW>']), Y

X, Y = readFile('USDRUB.csv')
n = len(Y)
trainLen = n / 2

X_train = X[:trainLen]
Y_train = Y[:trainLen]

X_test = X[trainLen:]
Y_test = Y[trainLen:]

clf = KNeighborsRegressor()
clf.fit(X_train, Y_train)

Y_test_predicted = clf.predict(X_test)

plt.plot(Y_test-Y_test_predicted)
plt.show()