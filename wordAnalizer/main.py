from os import walk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn import cross_validation
import sklearn
import numpy as np
import difflib

def readFile(fileName):
    dict = {}
    with open(fileName) as f:
        rows = f.readlines()
        for row in rows:
            components = row.strip().split('\t')
            secondComponent = components[1].split('+')
            dict[secondComponent[0]] = []

        for row in rows:
            components = row.strip().split('\t')
            secondComponent = components[1].split('+')

            dict[secondComponent[0]].append((components[0], secondComponent[1]))

    return dict

def difference(dict):

    for key in dict.keys():
        i = 0
        flag = True

        while (flag and len(dict[key][0][0]) > i):
            letter = dict[key][0][0][i]
            for word in dict[key]:
                if len(word[0]) == i or letter != word[0][i]:
                    flag = False
                    break
                i += 1

        print key, key[i:], i

def newDifference(dict):
    yTrain = []
    for key in dict.keys():
        val = dict[key]
        x = [difflib.SequenceMatcher(a=key, b=word[0]).get_matching_blocks()[0] for word in val]
        #get suffixes
        for i in xrange(len(x)):
            howManyCharactersAreEqual = x[i][2]
            maxString = val[i][0]
            tupleToAppend = (maxString[howManyCharactersAreEqual:], len(maxString)-howManyCharactersAreEqual)
            yTrain.append(tupleToAppend)
    print yTrain

newDifference(readFile('italian.txt.learn'))

#vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3,5))
