# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:24:20 2018

@author: sitandon
"""

from keras.models import model_from_json
import os
from sklearn import preprocessing
import pickle
import pandas as pd

folderPath = "F:\Sid\FL\Sentiment Analytics\Data"
testFile = os.path.join(folderPath,"Testing.csv")

positiveWords = pd.read_csv(os.path.join(folderPath,"PositiveWords.csv"),header = None, names = ["Words"])
negativeWords = pd.read_csv(os.path.join(folderPath,"NegativeWords.csv"),header = None, names = ["Words"])
neutralWords = pd.read_csv(os.path.join(folderPath,"NeutralWords.csv"),header = None, names = ["Words"])


testData = pd.read_csv(testFile,delimiter = '\t',names = ["Phrase"])
    
"""
# load json and create model
"""
json_file = open(os.path.join(folderPath,"model.json"), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join(folderPath,"model.h5"))

#--------Load objects
with open(os.path.join(folderPath,"objs.pkl"), "rb") as f:  # Python 3: open(..., 'rb')
    modelW2V,tfidf,n_dim,generateFeatureSet = pickle.load(f)

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
   
features = preprocessing.scale(
    generateFeatureSet(pd.DataFrame({"Reviews":testData.Phrase}),n_dim,modelW2V,tfidf)
    )
    
print(loaded_model.predict_classes(features))