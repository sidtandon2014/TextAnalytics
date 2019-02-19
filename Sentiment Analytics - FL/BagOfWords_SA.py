# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:30:14 2018

@author: sitandon
"""

from gensim.models import Word2Vec 
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import numpy as np
import string
import pandas as pd
import re

import sklearn.feature_extraction
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import text as txt
import nltk
from sklearn.metrics import classification_report

is5Classes = True

folderPath = "F:\Sid\FL\Sentiment Analytics\Data"
trainFile = os.path.join(folderPath,"train.tsv")
testFile = os.path.join(folderPath,"test.tsv")

trainData = pd.read_csv(trainFile,delimiter = '\t')
testData = pd.read_csv(testFile,delimiter = '\t')

positiveWords = pd.read_csv(os.path.join(folderPath,"PositiveWords.csv"),header = None, names = ["Words"])
negativeWords = pd.read_csv(os.path.join(folderPath,"NegativeWords.csv"),header = None, names = ["Words"])
neutralWords = pd.read_csv(os.path.join(folderPath,"NeutralWords.csv"),header = None, names = ["Words"])

punctuation = ["?",".","#","he","-",",",":","'s","\""]

"""
0: Somewhat NEgative
1: Negative
2: Neutral
3: Somewhat Positive
4: Postitive
"""

if(is5Classes): 
    #------Merge somewhat neagative and Negative
    trainData.loc[trainData.Sentiment == 0,'Sentiment'] = 1
    #--------Merge somewhat positive and Positive
    trainData.loc[trainData.Sentiment == 4,'Sentiment'] = 3

def allStopWords(reviews,minFrequency):
    dict = {}
    #tokens = []
    stopWords = list(txt.ENGLISH_STOP_WORDS)
    #myStopWords = txt.ENGLISH_STOP_WORDS.union(extraWords)
    for stmt in reviews:
        for word in stmt.split():
            lowerWord = word.lower()
            if lowerWord in dict:
                dict[lowerWord]+=1
            else:
                dict[lowerWord] = 1     
    for key, value in dict.items():
        if(value < minFrequency 
        & (key not in stopWords) 
        & (key not in neutralWords) 
        & (key not in negativeWords) 
        & (key not in positiveWords)):
            stopWords.append(key)
    return stopWords,dict

def removeStopWordsFromString(statement,myStopWords):
    sentence = []
    for word in statement.split():
        if (word.lower() not in myStopWords) & (len(word) > 1):
            sentence.append(word.lower())
    return sentence
    #return ' '.join(sentence)
    
#pd.Series(["1sdasd,1 2a?sdas sdf:,.\t's 100"]).apply(lambda x:' '.join(re.sub("([0-9])|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", x).split()))
    
def cleanFeeds(trainData, minFrequency,istraining = True):
    statements = []
    sentiments = []
    myStopWords,wordFreq = allStopWords(trainData.Phrase,minFreq)
    """
        Iterate over all the reviews 
        and remove empty reviews 
    """
    for index in trainData.index:
        review = trainData.iloc[index,].Phrase
        if istraining:
            sentimentVal = trainData.iloc[index,].Sentiment
        tmpStmt = removeStopWordsFromString(review,myStopWords)
        if len(tmpStmt) > 0:
            statements.append(tmpStmt)
            if istraining:
                sentiments.append(sentimentVal)
    return statements,sentiments,myStopWords,wordFreq



minFreq = 10
n_dim = 100

#---------Remove special characters
trainData.Phrase = trainData.Phrase.apply(
lambda x:' '.join(re.sub("([0-9])|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", x).split()))
     
reviews,sentiments,myStopWords,wordFreq = cleanFeeds(trainData,minFreq,True)
sentiments = np.array(sentiments)

len(reviews)
len(sentiments)
"""
import operator
sorted(wordFreq.iteritems(), key=operator.itemgetter(1))
"""

tfVectorizer = CountVectorizer(min_df = minFreq,stop_words = myStopWords)#,max_features = n_dim)
#-----Convert to dense array
tf_trainData_x = tfVectorizer.fit_transform(list(map(lambda x: ' '.join(x),reviews))).toarray()


train_Dataset = pd.DataFrame({"Reviews": list(map(lambda x: ' '.join(x),reviews)), "Sentiments" : sentiments})
train_Dataset.drop_duplicates(inplace = True)

tfVectorizer = CountVectorizer(min_df = minFreq,stop_words = myStopWords)
tf_trainData_x = tfVectorizer.fit_transform(train_Dataset.Reviews).toarray()

train_x,test_x,train_y,test_y = train_test_split(train_Dataset.Reviews,train_Dataset.Sentiments,test_size = 0.2)
np.unique(train_y)

train_y = train_y.reshape(train_y.shape[0],1) -1
test_y = test_y.reshape(test_y.shape[0],1) - 1


from sklearn import preprocessing 

train_x = preprocessing.scale(train_x)
test_x = preprocessing.scale(test_x)


from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras import losses
from keras.utils import to_categorical
from keras import regularizers
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

vocab_size = 7000
encoded_docs = [one_hot(d, vocab_size) for d in train_Dataset.Reviews]

max_length = 500
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
#print(padded_docs)

train_x,test_x,train_y,test_y = train_test_split(padded_docs,train_Dataset.Sentiments,test_size = 0.2)
train_y = train_y.reshape(train_y.shape[0],1) -1
test_y = test_y.reshape(test_y.shape[0],1) - 1
train_y = train_y.reshape(train_y.shape[0],1) -1
test_y = test_y.reshape(test_y.shape[0],1) - 1
train_keras_y = to_categorical(train_y, num_classes=3)
test_keras_y = to_categorical(test_y, num_classes=3)

kerasModel= Sequential()
    
embed_dim = 128
lstm_dim = 256
kerasModel.add(Embedding(vocab_size, embed_dim,input_length = max_length, dropout=0.2))

kerasModel.add(LSTM(lstm_dim, dropout_U=0.2, dropout_W=0.2, kernel_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l1(0.001)))

kerasModel.add(Dense(3,activation='softmax', kernel_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l1(0.001)))
              

kerasModel.compile(loss = 'categorical_crossentropy', optimizer='sgd',metrics = ['accuracy'])



#kerasModel.fit(train_x,train_keras_y,batch_size=32,epochs=5)   
kerasModel.fit(train_x,train_keras_y,batch_size=32,epochs=5)   

y_predict = kerasModel.predict_classes(test_x)
score = kerasModel.evaluate(test_x, test_keras_y, batch_size=128, verbose=2)
print(score[1])

report = classification_report(test_y,y_predict)

print(report)

trainData.groupby(["Sentiment"]).count()

