# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 23:06:12 2018

@author: sitandon
"""

import numpy as np
import pandas as pd

import sklearn.feature_extraction
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.cross_validation import train_test_split

is5Classes = True

folderPath = "F:\Sid\FL\Sentiment Analytics\Data"
trainFile = os.path.join(folderPath,"train.tsv")
testFile = os.path.join(folderPath,"test.tsv")

trainData = pd.read_csv(trainFile,delimiter = '\t')
testData = pd.read_csv(testFile,delimiter = '\t')

#----------check train data
trainData.head()
len(trainData)
"""
Data size: (156060)

"""

"""
Check whether data is skewed or not
"""
trainData.groupby(['Sentiment']).count()

if(is5Classes): 
    #------Merge somewhat neagative and Negative
    trainData.loc[trainData.Sentiment == 0,'Sentiment'] = 1
    #--------Merge somewhat positive and Positive
    trainData.loc[trainData.Sentiment == 4,'Sentiment'] = 3
    


"""
Data Cleansing

"""




countVectorizer = CountVectorizer()
#-----Convert to dense array
cv_trainData_x = countVectorizer.fit_transform(trainData.Phrase).toarray()
cv_trainData_x.shape
"""
Total features: (156060, 15240)
"""

featureHasher = FeatureHasher(input_type = 'string',n_features = 5000,non_negative = True)
fh_trainData_x = featureHasher.fit_transform(trainData.Phrase).toarray()
fh_trainData_x.shape

trainData_y = trainData.Sentiment.astype('category')
trainData_y.shape

#-----------Implement gaussianNB model
from sklearn.naive_bayes import GaussianNB

gaussianModel = GaussianNB()
gaussianModel.fit(fh_trainData_x,trainData_y)
gaussian_predict = gaussianModel.predict(fh_trainData_x)

(trainData_y == gaussian_predict).sum()/len(trainData)
"""
#--Observations for GAussian mdoel
1. Memory error with Count vectorizer
2. With FeatureHasher and 1000 features
    a) 1000 feature : 41% accuracy
    b) 5000 featuer : 39% accuracy
"""

#-----------Implement multinomialNB model
from sklearn.naive_bayes import MultinomialNB

multinomialModel = MultinomialNB()
multinomialModel.fit(fh_trainData_x,trainData_y)
multinomialModel_predict = multinomialModel.predict(fh_trainData_x)

(trainData_y == multinomialModel_predict).sum()/len(trainData)

"""
#--Observations for GAussian mdoel
1. Memory error with Count vectorizer
2. With FeatureHasher   
    a) 5000 feature : 51% accuracy
    
"""

"""
Implement KNN
"""

from sklearn.neighbors import KNeighborsClassifier


knnModel = KNeighborsClassifier(n_neighbors = 10)
knnModel.fit(fh_trainData_x,trainData_y)
knn_predict = knnModel.predict(fh_trainData_x)

(trainData_y == knn_predict).sum()/len(trainData)

"""
Observations KNN
Feature HAsher
    a) 5000 features: Not a good fit as the number of parameters are huge
"""

"""
Implement SVC
"""

from sklearn.svm import SVC

svcModel = SVC(kernel = 'linear',C=1)
svcModel.fit(cv_trainData_x,trainData_y)
svcModel.score(cv_trainData_x,trainData_y)  


    





