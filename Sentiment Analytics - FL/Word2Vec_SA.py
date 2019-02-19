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
from sklearn import preprocessing 

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
import pickle
from sklearn.metrics import classification_report


is5Classes = True
minFreq = 10
n_dim = 100

folderPath = "F:\Sid\FL\Sentiment Analytics\Data"
trainFile = os.path.join(folderPath,"train.tsv")
testFile = os.path.join(folderPath,"test.tsv")

trainData = pd.read_csv(trainFile,delimiter = '\t')
testData = pd.read_csv(testFile,delimiter = '\t')

positiveWords = pd.read_csv(os.path.join(folderPath,"PositiveWords.csv"),header = None, names = ["Words"])
negativeWords = pd.read_csv(os.path.join(folderPath,"NegativeWords.csv"),header = None, names = ["Words"])
neutralWords = pd.read_csv(os.path.join(folderPath,"NeutralWords.csv"),header = None, names = ["Words"])
stopWords = pd.read_csv(os.path.join(folderPath,"Stopwords.csv"),header = None, names = ["Words"])

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
    
    myStopWords = ["?",".","#","he","-",","] + list(stopWords.Words)
    for stmt in reviews:
        for word in stmt.split():
            lowerWord = word.lower()
            if lowerWord in dict:
                dict[lowerWord]+=1
            else:
                dict[lowerWord] = 1     
    for key, value in dict.items():
        if(value < minFrequency 
        & (key not in myStopWords) 
        & (key not in neutralWords) 
        & (key not in negativeWords) 
        & (key not in positiveWords)):
            myStopWords.append(key)
    return myStopWords

def removeStopWordsFromString(statement,myStopWords):
    sentence = []
    for word in statement.split():
        if word.lower() not in myStopWords:
            sentence.append(word.lower())
    return sentence
    #return ' '.join(sentence)
    
def cleanFeeds(data, minFrequency,myStopWords,isTraining = True):
    statements = []
    sentiments = []
    #---------Remove special characters
    data.Phrase = data.Phrase.apply(
    lambda x:' '.join(re.sub("([0-9])|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", x).split()))
    
    """
        Iterate over all the reviews 
        and remove empty reviews 
    """
    for index in data.index:
        review = data.iloc[index,].Phrase
        if isTraining:
            sentimentVal = trainData.iloc[index,].Sentiment
        tmpStmt = removeStopWordsFromString(review,myStopWords)
        if len(tmpStmt) > 0:
            statements.append(tmpStmt)
            if isTraining:
                sentiments.append(sentimentVal)
    return statements,sentiments

def generateTokens(data,data_Full,n_dim,myStopWords):
    phrases = Phrases(data)
    biggram = Phraser(phrases)
    """
    #---------Check multiple token for a word
    biggram = Phraser(phrases)
    biggram[reviews[0]]
    
    """
    #lstReviews = list(train_Dataset.Reviews.apply(lambda x: get_bigrams(x)))
    modelW2V = Word2Vec(biggram[data]
                        ,size = n_dim
                        ,min_count = minFreq
                        ,window = 2
                        ,sg = 1)        
    
    """
    Term document frequency for weighted average of features
    """
    
    tfVectorizer = TfidfVectorizer(min_df = minFreq,ngram_range = (1,2),stop_words = myStopWords,max_features = n_dim)
    tfVectorizer.fit(data_Full)
    tfFeatureSet = tfVectorizer.transform(data_Full)
    
    tfidf = dict(zip(
                list(map(lambda x: str.replace(x," ","_"),tfVectorizer.get_feature_names()))
                , tfVectorizer.idf_)
            )
    return modelW2V,tfidf,tfFeatureSet

def get_bigrams(myString):

    from nltk.stem import PorterStemmer
    from nltk.tokenize import WordPunctTokenizer
    from nltk.collocations import BigramCollocationFinder
    from nltk.metrics import BigramAssocMeasures
    try:
        tokenizer = WordPunctTokenizer()
        tokens = tokenizer.tokenize(myString)
        bigram_finder = BigramCollocationFinder.from_words(tokens)
        bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)

        for bigram_tuple in bigrams:
            x = "%s_%s" % bigram_tuple
            tokens.append(x)
    except ZeroDivisionError:
        print(myString)
        #pdb.set_trace()
    return tokens        
        
#-------Generate feature set
def generateFeatureSet(data,n_dim,modelW2V,tfidfMatrix):    
    featureSet = np.zeros((len(data.Reviews),n_dim))
    wordCount = np.zeros((len(data.Reviews),3))
    errorWords = []
    for index,stmt in enumerate(data.Reviews):
        vec = np.zeros((1,n_dim))
        count = 0
        countPos = 0
        countNeg = 0
        countNeutral = 0
        """
        Generate list of unigrams and bigrams 
        """
        for word in get_bigrams(stmt):
            try:
                #vec += modelW2V[word].reshape(1,n_dim) * tfidfMatrix[word]
                vec += modelW2V[word].reshape(1,n_dim) * math.sqrt(tfidfMatrix[word])
                if word in positiveWords:
                    countPos+=1
                elif word in negativeWords:
                    countNeg+=1
                elif word in neutralWords:
                    countNeutral +=1
            except KeyError:
                errorWords.append(word)
                continue    
            count+=1
        if count!=0:
            vec = vec/count            
            wordCount[index,0] = countPos
            wordCount[index,1] = countNeg
            wordCount[index,2] = countNeutral            
        featureSet[index] = vec
    return np.concatenate((featureSet,wordCount),axis = 1)

#featureSet = np.delete(featureSet,dropIndex,0)
#sentiments = np.array(sentiments)
#sentiments = np.delete(sentiments,dropIndex,0)

#len(train_Dataset.Sentiments)
#featureSet.shape

myStopWords = allStopWords(pd.concat([trainData.Phrase,testData.Phrase]),minFreq)
reviews,sentiments = cleanFeeds(trainData,minFreq,myStopWords,True)
"""
Create dataframe
"""

train_Dataset = pd.DataFrame({"Reviews": list(map(lambda x: ' '.join(x),reviews)), "Sentiments" : sentiments})
train_Dataset.drop_duplicates(inplace = True)

"""
Take miimum of segment value in case there are same reviews but different sentiment
"""

train_Dataset = pd.DataFrame(train_Dataset.groupby("Reviews",as_index = False)["Sentiments"].min())
train_Dataset.to_csv(os.path.join(folderPath,"Dataset.csv"))

"""
Generate tokens and Word2Vec model
"""
#tmp = list(train_Dataset.Reviews.apply(lambda x: x.split()))
modelW2V,tfidf,tfFeatureSet = generateTokens(
            list(train_Dataset.Reviews.apply(lambda x: x.split()))
            ,list(train_Dataset.Reviews)            
            ,n_dim
            ,myStopWords)
"""
write keys to csv
with open(os.path.join(folderPath,"Keys.csv"), 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in modelW2V.wv.vocab.items()]
"""

featureSet = generateFeatureSet(train_Dataset,n_dim,modelW2V,tfidfMatrix=tfidf)

"""
Split the data into training and testing set
"""

train_x,test_x,train_y,test_y = train_test_split(featureSet,train_Dataset.Sentiments,test_size = 0.2)
#np.unique(train_y)

train_y = train_y.reshape(train_y.shape[0],1) -1
test_y = test_y.reshape(test_y.shape[0],1) - 1
train_x = preprocessing.scale(train_x)
test_x = preprocessing.scale(test_x)

"""
Random Forest classifier
47%
"""
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_features=100)
rfc.fit(train_x,train_y)
rfc.score(test_x,test_y)
print(classification_report(test_y,rfc.predict(test_x)))

"""
Ada Boost: taking lot of time while training
"""
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier() 
clf = AdaBoostClassifier(n_estimators=50, base_estimator=dt,learning_rate=1)
clf.fit(train_x,train_y)
clf.score(test_x,test_y)
classification_report(test_y,clf.predict(test_x))

"""
Gradient Boost: try it out
"""
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
gbc.fit(train_x,train_y)
gbc.score(test_x,test_y)
classification_report(test_y,clf.predict(test_x))

"""
XGBoost
"""
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
xgb.fit(train_x,train_y)
xgb.score(test_x,test_y)
classification_report(test_y,clf.predict(test_x))

"""
Decision trees:50%

"""
from sklearn.tree import DecisionTreeClassifier

modelDT = DecisionTreeClassifier(max_depth = 5)
modelDT.fit(train_x,train_y)
y_predict = modelDT.predict(test_x)
print(classification_report(test_y,y_predict))
modelDT.score(test_x,test_y)

"""
Logistic Regression
Accuracy: 55% with newton-cg
"""
from sklearn.linear_model import LogisticRegression as LR

logitmodel = LR(multi_class = 'multinomial',solver ='newton-cg')
logitmodel.fit_transform(train_x,train_y)
logitmodel.score(test_x,test_y)


"""
SVC taking lot of time while training
"""
from sklearn.svm import SVC

svcModel = SVC(kernel = 'rbf',C=1)
svcModel.fit(train_x,train_y)
svcModel.score(test_x,test_y)  

"""
Naive Bayes
"""
#-----------Implement multinomialNB model
from sklearn.naive_bayes import MultinomialNB

#---------Check GAussian behaviour of each feature
multinomialModel = MultinomialNB()
multinomialModel.fit(train_x,train_y)
#multinomialModel_predict = multinomialModel.predict(te)
multinomialModel.score(test_x,test_y)
classification_report(test_y,multinomialModel.predict(test_x))

"""
Voting Classifier
"""

from sklearn.ensemble import VotingClassifier

estimators = []
model1 = MultinomialNB()
estimators.append(('MultinomalNB', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(train_x, train_y)
classification_report(test_y,ensemble.predict(test_x))


"""
Keras model:
    Accuracy: 62%
"""

from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout,LSTM
from keras import losses
from keras.utils import to_categorical

kerasModel= Sequential()

kerasModel.add(Dense(units=32, activation='relu', input_dim=n_dim + 3))
kerasModel.add(Dropout(0.2))
kerasModel.add(Dense(units=64, activation='relu'))
kerasModel.add(Dropout(0.2))
kerasModel.add(Dense(units=3, activation='softmax'))

kerasModel.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
              


train_keras_y = to_categorical(train_y, num_classes=3)
test_keras_y = to_categorical(test_y, num_classes=3)

kerasModel.fit(train_x,train_keras_y,batch_size=32,epochs=30)   

y_predict = kerasModel.predict_classes(test_x)
score = kerasModel.evaluate(test_x, test_keras_y, batch_size=128, verbose=2)
print(score[1])

report = classification_report(test_y,y_predict)
print(report)

#train_Dataset.groupby(["Sentiments"]).count()

test = [
"IT was an ok movie",
"good movie",
"awesome",
"was not a good movie",
"boring",
"disgusting movie",
"The sun rises in the east",
"I hate the book"
]

features = preprocessing.scale(
    generateFeatureSet(pd.DataFrame({"Reviews":test}),n_dim,modelW2V,tfidf)
    )

print(kerasModel.predict_classes(features))


"""
Save model to disk with all other objects
"""
# serialize model to JSON
model_json = kerasModel.to_json()
with open(os.path.join(folderPath,"model.json"), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
kerasModel.save_weights(os.path.join(folderPath,"model.h5"))

with open(os.path.join(folderPath,"objs.pkl"), "wb") as f:  # Python 3: open(..., 'wb')
    pickle.dump([modelW2V,tfidf,n_dim,generateFeatureSet], f)

print("Saved model to disk")

           
tfVectorizer1 = TfidfVectorizer(ngram_range = (1,2),max_features = 10)
tfVectorizer1.fit(["Sameer dsd sdasdf dfsdfsd sdfsdfsdf"])
test = tfVectorizer1.transform(["Sameer dsd sdasdf dfsdfsd sdfsdfsdf"])
test.toarray()