import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split



class TagPredictor:
    classifier = None
    model = None
    corpus = None

    def __init__(self, classifier, corpus):
        self.classifier = classifier
        self.corpus = corpus

        np.random.seed(500)

        print("Initialized TagPredictor")
        
        
    def train(self):
        print("Started training")
        
        # Transform tags to multilabel format
        self.mlb = MultiLabelBinarizer()
        Y_matrix = self.mlb.fit_transform(self.corpus['Tags'])
        #np.set_printoptions(threshold=np.inf)
        #print(matrix[0])
        print(self.mlb.classes_)
        train, test, Train_Y, Test_Y = train_test_split(self.corpus, Y_matrix, test_size=0.3, shuffle=True)
        Train_X = train['Bag_of_Words']
        Test_X = test['Bag_of_Words']
        #print(Train_X)
        #print(Train_Y)

        self.Tfidf_vect = TfidfVectorizer(max_features=5000)
        self.Tfidf_vect.fit(self.corpus['Bag_of_Words'])
        Train_X_Tfidf = self.Tfidf_vect.transform(Train_X)
        Test_X_Tfidf = self.Tfidf_vect.transform(Test_X)

        self.model = self.classifier()
        self.model.train(Train_X_Tfidf, Train_Y)
        
        print("Finished training")
        
        
    def predict(self, df):
        # return predictions_df, confidence_level
        X = df
        X_Tfidf = self.Tfidf_vect.transform(X)
        matrix, confidenceList = self.model.predict(X_Tfidf)
        labels = self.mlb.inverse_transform(matrix)
        return labels, confidenceList