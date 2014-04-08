import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#from sklearn import cross_validation
print "reading files..."
from numpy import genfromtxt
trainFile = genfromtxt(open("./train.csv",'r'), delimiter = ',')[1:]
testFile = genfromtxt(open("./test.csv",'r'), delimiter = ',')[1:]

#traintmp = np.zeros(shape=trainFile.shape)
##train and test rom RGB to CbCr
#for i,x in enumerate(trainFile):
#    train[i,0]=(0.299*x[0]+0.587*x[1]+0.114*x[2])
#    train[i,1]=128-0.168736*x[0]-0.331264*x[1]+0.5*x[2]
#    train[i,2]=128+0.5*x[0]-0.418688*x[1]-0.081312*x[2]
#    train[i,3]=x[3]

from collections import Counter

print "testing..."
from sklearn.neighbors import NearestNeighbors
n = 25
NN = NearestNeighbors(n_neighbors = n)
NN.fit(trainFile[:,:3])
tmp = []
for y in trainFile:
    nearest = NN.kneighbors(y[:3])
    testnn = np.asarray(nearest)
    gn = 0.
    gun = 0.
    for x in testnn[1,0].astype(int):    
        if trainFile[x,3] == y[3]:
            gn += 1
        else:
            gun += 1 
    margin = (gn - gun)/n
    if (margin > -0.6) or (y[3]==-1):
        tmp.append(y)
trainFileIm = np.asarray(tmp)
#trainFileIm = trainFile
print trainFile.shape
print trainFileIm.shape
train = np.zeros(shape=trainFileIm.shape)
test = np.zeros(shape=testFile.shape)
#train and test rom RGB to CbCr
for i,x in enumerate(trainFileIm):
    train[i,0]=(0.299*x[0]+0.587*x[1]+0.114*x[2])
    train[i,1]=128-0.168736*x[0]-0.331264*x[1]+0.5*x[2]
    train[i,2]=128+0.5*x[0]-0.418688*x[1]-0.081312*x[2]
    train[i,3]=x[3]
for i,x in enumerate(testFile):
    test[i,0]=x[0]
    test[i,1]=(0.299*x[1]+0.587*x[2]+0.114*x[3])
    test[i,2]=128-0.168736*x[1]-0.331264*x[2]+0.5*x[3]
    test[i,3]=128+0.5*x[1]-0.418688*x[2]-0.081312*x[3]

#train = np.zeros(shape=trainfile.shape)
#np.savetxt("C:/Users/demist/Documents/MIPT/ML/skin/newtrain.csv",train,delimiter=',',fmt='%g')

trainF = trainFileIm[:,:3]
trainA = trainFileIm[:,3]
testF = test[:,1:]
testRGB = testFile[:,1:]
#from mpl_toolkits.mplot3d import Axes3D

#for 3 different KNN's
trainXY = train[:,[0,1]];
trainYZ = train[:,[1,2]];
trainXZ = train[:,[0,2]];
RGBtrainXY = trainFileIm[:,[0,1]];
RGBtrainYZ = trainFileIm[:,[1,2]];
RGBtrainXZ = trainFileIm[:,[0,2]];
#ReShaping classes

#to center
#trainXY[:,0] -= 128
#trainYZ[:,0] -= 128
#trainXZ[:,0] -= 128
#trainXY[:,1] -= 128
#trainYZ[:,1] -= 128
#trainXZ[:,1] -= 128
##normalize
#trainXY[:,0] *= 128
#trainYZ[:,0] *= 128
#trainXZ[:,0] *= 128
#trainXY[:,1] *= 128
#trainYZ[:,1] *= 128
#trainXZ[:,1] *= 128

#3D plot
#fig = plt.figure();
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(xs=train[:,0],ys=train[:,1],zs=train[:,2],c=train[:,3]);
#plt.show();

trainXYst = []
trainYZst = []
trainXZst = []
for y,x in enumerate(train):
    if x[3]==-1:
        trainXYst.append((x[0],x[1]))
        trainYZst.append((x[1],x[2]))
        trainXZst.append((x[0],x[2]))
trainXYs = np.asarray(trainXYst)
trainYZs = np.asarray(trainYZst)
trainXZs = np.asarray(trainXZst)
print "plotting..."
#2D plots
#fig = plt.figure();
#ax = fig.add_subplot(221);
#ax.scatter(x=trainXYs[:,0],y=trainXYs[:,1],s=3)
#ax.set_title("XY");
#ax.set_autoscaley_on(False);
#ax.set_xlim([0,256]);
#ax.set_ylim([0,256]);
#ax = fig.add_subplot(222);
#ax.scatter(x=trainYZs[:,0],y=trainYZs[:,1],s=3)
#ax.set_title("YZ");
#ax.set_autoscaley_on(False);
#ax.set_xlim([0,256]);
#ax.set_ylim([0,256]);
#ax = fig.add_subplot(223);
#ax.scatter(x=trainXZs[:,0],y=trainXZs[:,1],s=3)
#ax.set_title("XZ");
#ax.set_autoscaley_on(False);
#ax.set_xlim([0,256]);
#ax.set_ylim([0,256]);
#plt.show();

#cross validation
#from sklearn.cross_validation import train_test_split
#trainCV, testCV = train_test_split(train,test_size=0.33,random_state=42)
#trainF = trainCV[:,:3]
#trainA = trainCV[:,3]
#testF = testCV[:,:3]
#testA = testCV[:,3]

#simple CV
#clf = KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',n_neighbors=3,p=1,weights='uniform')
#clf.fit(trainF,trainA)
#scores = cross_validation.cross_val_score(clf,train[:,:3],train[:,3],cv=5)
#print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

#K-Fold
#cv = cross_validation.KFold(len(trainA),n_folds=5,shuffle = True,indices=False)
#results = []
#clf = KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',n_neighbors=3,p=2,weights='uniform')
#for traincv, testcv in cv:
#    results.append(clf.fit(trainF[traincv], trainA[traincv]).score(trainF[testcv],trainA[testcv]))
#res = np.asarray(results)
#print np.mean(res)

#grid search
print "grid search cv..."
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
parameters = {'n_neighbors':[5,7,9,11,13,15],'weights':['uniform','distance']}
clf1 = GridSearchCV(KNeighborsRegressor(), parameters, cv = 5)
clf1.fit(trainF,trainA)
#print "XY"
print clf1.best_estimator_
#parameters = {'n_neighbors':[5,7,9,11,13,15],'weights':('uniform','distance')}
#clf2 = GridSearchCV(KNeighborsRegressor(),parameters,cv = 5)
#clf2.fit(RGBtrainYZ,trainA)
#print "YZ"
#print clf2.best_estimator_
#parameters = {'n_neighbors':[5,7,9,11,13,15],'weights':('uniform','distance')}
#clf3 = GridSearchCV(KNeighborsRegressor(),parameters,cv = 5)
#clf3.fit(RGBtrainXZ,trainA)
#print "XZ"
#print clf3.best_estimator_

#knn
print "3D knn..."
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
knn3D1 = KNeighborsRegressor(metric='minkowski',n_neighbors=5,p=2,weights='distance')
knn3D2 = KNeighborsRegressor(metric='minkowski',n_neighbors=3,p=1,weights='uniform')
knn3D1.fit(trainF,trainA)
knn3D2.fit(trainF,trainA)
#result3D1 = knn3D1.predict(testF)
#result3D2 = knn3D2.predict(testF)
#import numpy as np
#result = knn.predict(testF)

print "3KNNs..."
#3 KNN's for different planes
from sklearn.neighbors import KNeighborsClassifier
#initializing
knnXY = KNeighborsClassifier(metric='minkowski',n_neighbors=5,p=2,weights='uniform')
knnYZ = KNeighborsClassifier(metric='minkowski',n_neighbors=5,p=2,weights='uniform')
knnXZ = KNeighborsClassifier(metric='minkowski',n_neighbors=5,p=2,weights='uniform')
RGBknnXY = KNeighborsRegressor(n_neighbors=5,weights='uniform') #5
RGBknnYZ = KNeighborsRegressor(n_neighbors=11,weights='uniform') #11
RGBknnXZ = KNeighborsRegressor(n_neighbors=7,weights='uniform') #7

#fitting
knnXY.fit(trainXY,trainA)
knnYZ.fit(trainYZ,trainA)
knnXZ.fit(trainXZ,trainA)
RGBknnXY.fit(RGBtrainXY,trainA)
RGBknnYZ.fit(RGBtrainYZ,trainA)
RGBknnXZ.fit(RGBtrainXZ,trainA)

#print trainYZ.shape
#cooking Image
#readBMP
#from PIL import Image
#img = Image.open("C:/Users/demist/Documents/MIPT/ML/skin/img.bmp")
#image = np.array(img.getdata(),np.uint8).reshape(img.size[1],img.size[0],3)
#temp = []
#for i,v in enumerate(image):
#    for j,t in enumerate(v):
#        temp.append(t)
#test = np.asarray(temp)          
#print test.shape

#test for image
#testXY = test[:,[0,1]]
#testYZ = test[:,[1,2]]
#testXZ = test[:,[0,2]]

#cooking test for KNN's
testXY = test[:,[1,2]]
testYZ = test[:,[2,3]]
testXZ = test[:,[1,3]]
RGBtestXY = testFile[:,[1,2]]
RGBtestYZ = testFile[:,[2,3]]
RGBtestXZ = testFile[:,[1,3]]


#ReShaping

#to center
#testXY[:,0] -= 128
#testYZ[:,0] -= 128
#testXZ[:,0] -= 128
#testXY[:,1] -= 128
#testYZ[:,1] -= 128
#testXZ[:,1] -= 128

#normalize
#testXY[:,0] *= 128
#testYZ[:,0] *= 128
#testXZ[:,0] *= 128
#testXY[:,1] *= 128
#testYZ[:,1] *= 128
#testXZ[:,1] *= 128

#print testXY.shape 
#print trainXY.shape
#print testYZ.shape
#print trainYZ.shape
#print testXZ.shape
#print trainXZ.shape
#print test[:,:].shape
#print trainF.shape

#predicting 
resultXY = knnXY.predict(testXY)
resultYZ = knnYZ.predict(testYZ)
resultXZ = knnXZ.predict(testXZ)
result3D1 = knn3D1.predict(testRGB)
result3D2 = knn3D2.predict(testRGB)
RGBresultXY = RGBknnXY.predict(RGBtestXY)
RGBresultYZ = RGBknnYZ.predict(RGBtestYZ)
RGBresultXZ = RGBknnXZ.predict(RGBtestXZ)
#print resultXY.shape
#print resultYZ.shape
#print resultXZ.shape
#print result3D1.shape
#print result3D2.shape

#print "stacking..."
##stacking fit
#trXY = knnXY.predict(trainXY);
#trYZ = knnYZ.predict(trainYZ);
#trXZ = knnXZ.predict(trainXZ);

#knnStack =  KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',n_neighbors=3,p=1,weights='distance')
#knnStack.fit(np.column_stack((trXY,trYZ,trXZ)),trainA);

##stacking
#result = knnStack.predict(np.column_stack((resultXY,resultYZ,resultXZ)))

#from sklearn.neighbors import KNeighborsRegressor
#n = KNeighborsRegressor(n_neighbors = 3)
#n.fit(trainF,trainA)
#r1 = n.predict(testRGB)

#from sklearn import linear_model 
#logXY = linear_model.LinearRegression()
#logYZ = linear_model.LogisticRegression()
#logXZ = linear_model.LogisticRegression()
#logXY.fit(trainF,trainA)
#logYZ.fit(RGBtrainYZ,trainA)
#logXZ.fit(RGBtrainXZ,trainA)
#r1 = logXY.predict(testRGB)
#r2 = logYZ.predict(RGBtestYZ)
#r3 = logXZ.predict(RGBtestXZ)
#resulttmp = r1 + r2 + r3
#resulttmp = r1
print "voting..."
#voting
resulttmp = result3D1
result = np.zeros(shape=resulttmp.shape)
for i,v in enumerate(resulttmp):
    if v<0:
        result[i]=-1
    else:
        result[i]=1
#check = 0

#SVM
#print "SVM..."
#from sklearn.svm import SVC
#clf = SVC();
#clf.fit(trainF,trainA)
#resultSVM = clf.predict(testF)
#trainExt = train
#for x,v in enumerate(result):
#    c = result3D1[x]+result3D2[x]+resultXY[x]+resultXZ[x]+resultYZ[x]
#    if  c == 5:
#        tmp = np.column_stack((testF[x,0],testF[x,1],testF[x,2],v))
#        np.concatenate((trainExt,tmp))
        #check = check + 1
#    if  c == -5:
#        tmp = np.column_stack((testF[x,0],testF[x,1],testF[x,2],v))
#        np.concatenate((trainExt,tmp))
        #check = check + 1
#everything with appended train
#trainFext = trainExt[:,:3]
#trainAext = trainExt[:,3]
#trainXYext = trainExt[:,[0,1]];
#trainYZext = trainExt[:,[1,2]];
#trainXZext = trainExt[:,[0,2]];
#print "Second KNN set..."
#from sklearn.neighbors import KNeighborsClassifier
#initializing
#knnXYext = KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',n_neighbors=3,p=2,weights='uniform')
#knnYZext = KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',n_neighbors=3,p=2,weights='uniform')
#knnXZext = KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',n_neighbors=3,p=2,weights='uniform')
#knn3D1ext = KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',n_neighbors=3,p=2,weights='uniform')
#knn3D2ext = KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',n_neighbors=3,p=1,weights='uniform')

#fitting
#knnXYext.fit(trainXYext,trainAext)
#knnYZext.fit(trainYZext,trainAext)
#knnXZext.fit(trainXZext,trainAext)
#knn3D1ext.fit(trainFext,trainAext)
#knn3D2ext.fit(trainFext,trainAext)

#predicting 
#resultXYext = knnXYext.predict(testXY)
#resultYZext = knnYZext.predict(testYZ)
#resultXZext = knnXZext.predict(testXZ)
#result3D1ext = knn3D1ext.predict(testF)
#result3D2ext = knn3D2ext.predict(testF)

#voting
#resulttmpext = resultXYext + resultYZext + resultXZext + result3D1ext + result3D2ext;
#resultext = np.zeros(shape=resulttmpext.shape)
#for i,v in enumerate(resulttmpext):
#   if v<0:
#        resultext[i]=-1
#    else:
#        resultext[i]=1
#
#making out BMP
#resultImg = np.zeros(image.shape, dtype = np.uint8)
#for i,x in enumerate(image):
#    for j,v in enumerate(x):
#        if result[j+i*img.size[0]]==1:
#            resultImg[i,j] = [0,255,0]
#        else:
#            resultImg[i,j] = v

#np.savetxt("C:/Users/demist/Documents/MIPT/ML/skin/image.csv",image,delimiter=',',fmt='%g')
#np.savetxt("C:/Users/demist/Documents/MIPT/ML/skin/resImg.csv",resultImg,delimiter=',',fmt='%g')
#np.savetxt("C:/Users/demist/Documents/MIPT/ML/skin/resIMAGE.csv",result,delimiter=',',fmt='%g')

#img2 = Image.fromarray(resultImg)
#img2.save("C:/Users/demist/Documents/MIPT/ML/skin/imgSKIN.bmp")

testXYst = []
testYZst = []
testXZst = []
for y,x in enumerate(test):
    if result[y]==1:
        testXYst.append((x[1],x[2]))
        testYZst.append((x[2],x[3]))
        testXZst.append((x[1],x[3]))
testXYs = np.asarray(testXYst)
testYZs = np.asarray(testYZst)
testXZs = np.asarray(testXZst)
#2D plots for result
fig = plt.figure();
#ax = fig.add_subplot(221);
#ax.scatter(x=testXY[:,0],y=testXY[:,1],c=result)
#ax.set_title("XY");
#ax.set_autoscaley_on(False);
#ax.set_xlim([0,256]);
#ax.set_ylim([0,256]);
ax = fig.add_subplot(111);
ax.scatter(x=testYZs[:,0],y=testYZs[:,1],s=5)
ax.set_title("CbCr");
ax.set_autoscaley_on(False);
ax.set_xlim([0,256]);
ax.set_ylim([0,256]);
#ax = fig.add_subplot(223);
#ax.scatter(x=testXZ[:,0],y=testXZ[:,1],c=result)
#ax.set_title("XZ");
#ax.set_autoscaley_on(False);
#ax.set_xlim([0,256]);
#ax.set_ylim([0,256]);
plt.show();

print "outputing..."
np.savetxt("C:/Users/demist/Documents/MIPT/ML/skin/res3KNN.csv",np.column_stack((test[:,0],result)),delimiter=',',fmt='%g')
#np.savetxt("C:/Users/demist/Documents/MIPT/ML/skin/3knnvotes.csv",np.column_stack((resultXY,resultYZ,resultXZ,result)),delimiter=',',fmt='%g')
#output
#np.savetxt("C:/Users/demist/Documents/MIPT/ML/skin/res.csv",np.column_stack((test[:,0],result)),delimiter=',',fmt='%g')
