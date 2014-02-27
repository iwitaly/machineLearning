import numpy as np
from sklearn import cross_validation, datasets, svm
from pylab import *
import matplotlib.pyplot as plt

digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)

scores = []

for C in C_s:
    svc.C = C
    buffScores = cross_validation.cross_val_score(svc, X, y)
    scores.append(sum(buffScores) / len(buffScores))

import pylab as pl
pl.figure(1, figsize=(4, 3))
pl.clf()
pl.semilogx(C_s, scores)
#locs, labels = pl.yticks()
#pl.yticks(locs, map(lambda x: "%g" % x, locs))
pl.ylabel('CV score')
pl.xlabel('Parameter C')
pl.ylim(0, 1.1)
pl.show()