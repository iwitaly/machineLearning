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

def readFile():
    dict = {}
    with open('italian.txt.learn') as f:
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

    print dict.items()[0]

    for key in dict.keys():
        i = 0
        flag = True

        while flag and len(dict[key][0][0]) > i:

            letter = dict[key][0][0][i]
            for word in dict[key]:
                if len(word[0]) == i or letter != word[0][i]:
                    flag = False
                    break
                i += 1

        print key, key[i:], i

readFile()
difference(readFile())

vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3,5))
