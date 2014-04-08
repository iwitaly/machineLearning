from os import walk
from sys import argv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.decomposition import PCA
import sklearn
import numpy as np
import matplotlib.pyplot as plt


def readTexts(dirpath):
	textList = []
	for root, dirs, files in walk(dirpath):
	    for file in files:
			fin = open(dirpath + '/' + file, 'r')
			text = ""
			for line in fin.readlines():
				text += unicode(line, errors='replace')
			textList.append(text)
			fin.close()
	return textList

sportCorpus = readTexts("rec.sport.hockey")
politCorpus = readTexts("talk.politics.mideast")
compCorpus = readTexts("comp.graphics")

labels = ['sport'] * len(sportCorpus) + ['polit'] * len(politCorpus) + ['comp'] * len(compCorpus)

corpus = sportCorpus + politCorpus + compCorpus

def predictWithClassifier(model=RandomForestClassifier()):
    kfold = cross_validation.KFold(len(corpus), n_folds=5)
    results = []

    vectorizer = CountVectorizer(max_features=1000)
    p = PCA(n_components=20)

    for train_indices, test_indices in kfold:
        featureList = vectorizer.fit_transform([corpus[i] for i in train_indices])
        featureList = p.fit_transform(featureList.toarray())

        model.fit(featureList, [labels[i] for i in train_indices])

        featureListTest = vectorizer.transform([corpus[i] for i in test_indices])
        featureListTest = p.transform(featureListTest.toarray())

        results.append(model.score(featureListTest, [labels[i] for i in test_indices]))

    print sum(results) / len(results)

def plotText(): #create featured description for all train_set and plot points with different color for different texts
    vectorizer = CountVectorizer(max_features=1000)
    p = PCA(n_components=2)

    c = vectorizer.fit_transform(corpus)
    c = p.fit_transform(c.toarray())
    x, y = [], []

    for i in c:
        x.append(i[0])
        y.append(i[1])

    n1, n2 = len(sportCorpus), len(politCorpus)

    plt.plot(x[:n1], y[:n1], 'ro', x[n1:n1+n2], y[n1:n1+n2], 'go', x[n1+n2:], y[n1+n2:], 'bo')
    plt.show()

#predictWithClassifier()

plotText()