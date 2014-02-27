from os import walk
from sys import argv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn import cross_validation
import sklearn
import numpy as np

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


vectorizer = CountVectorizer(min_df = 1)

sportCorpus = readTexts("rec.sport.hockey.hockey")
politCorpus = readTexts("talk.politics.mideast")
compCorpus = readTexts("comp.graphics")

labels = ['sport'] * len(sportCorpus) + ['polit'] * len(politCorpus) + ['comp'] * len(compCorpus)

corpus = sportCorpus + politCorpus + compCorpus

model = LinearSVC()

#kfold = cross_validation.KFold(len(corpus), n_folds=10)
kfold = cross_validation.StratifiedKFold(corpus, n_folds=4)
results = []

for train_indices, test_indices in kfold:
    featureList = vectorizer.fit_transform([corpus[i] for i in train_indices])
    model.fit(featureList, [labels[i] for i in train_indices])

    featureListTest = vectorizer.transform([corpus[i] for i in test_indices])
    results.append(model.score(featureListTest, [labels[i] for i in test_indices]))

print sum(results) / len(results)