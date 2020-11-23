'''
@file       TagPredictor.py
@date       2020/08/25
@brief      Tag prediction ML module in the active learning architecture
'''

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


'''
Tag prediction ML module in the active learning architecture
'''
class TagPredictor:
    model = None            # Scikit-learn like classifier object
    database = None         # Pandas dataframe of topic text and tags


    '''
    @brief      Class constructor
    @param      classifier  Scikit-learn like classifier object
    @param      database    Pandas dataframe of topic text and tags
    @return     None
    '''
    def __init__(self, classifier, database):
        # Instantiate classifier object and set database
        self.model = classifier()
        self.database = database

        np.random.seed(500)

        #print("Initialized TagPredictor")
        

    '''
    @brief      Trains the model using the database
    @param      None
    @return     None
    '''
    def train(self):
        print("Started training")
        
        # Transform tags to multilabel indicator matrix
        self.mlb = MultiLabelBinarizer()
        Y_matrix = self.mlb.fit_transform(self.database['Tags'])
        
        #np.set_printoptions(threshold=np.inf)
        #print(matrix[0])
        print(self.mlb.classes_)
        
        #train, test, Train_Y, Test_Y = train_test_split(self.database, Y_matrix, test_size=0.3, shuffle=True)
        #Train_X = train['Bag_of_Words']
        #Test_X = test['Bag_of_Words']
        #print(Train_X)
        #print(Train_Y)
        
        # Create TF-IDF vectorizer from the database
        self.Tfidf_vect = TfidfVectorizer(max_features=5000)
        self.Tfidf_vect.fit(self.database['Bag_of_Words'])
        
        # Vectorize topic bags of words
        Train_X_Tfidf = self.Tfidf_vect.transform(self.database['Bag_of_Words'])
        #Test_X_Tfidf = self.Tfidf_vect.transform(Test_X)
        # Train model
        self.model.train(Train_X_Tfidf, Y_matrix)
        
        print("Finished training")


    '''
    @brief      Predicts a list of tags and confidence level for each topic
    @param      df                  Pandas dataframe of topic text
    @return     labels              List of predicted tuples of tags for each topic
    @return     confidenceList      NumPy array of prediction confidence scores for each topic
    '''    
    def predict(self, df):
        # Vectorize topic bags of words
        X_Tfidf = self.Tfidf_vect.transform(df)
        
        # Get the prediction matrix and confidence list
        predictionMatrix, confidenceList = self.model.predict(X_Tfidf)
        
        # Map the prediction indicator matrix back to lists of labels (tags)
        labels = self.mlb.inverse_transform(predictionMatrix)
        
        return labels, confidenceList
