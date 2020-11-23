import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from classifier import Classifier

class Classifier_SVM(Classifier):
    kernel = None
    model = None

    def __init__(self, kernel='linear'):
        self.kernel = kernel

    def train(self, Train_X_Tfidf, Train_Y, Test_X_Tfidf=None, Test_Y=None):
        print("Running SVM Classifier")

        # Classifier - Algorithm - SVM
        # fit the training dataset on the classifier
        self.model = OneVsRestClassifier(SVC(C=1.0, kernel=self.kernel, degree=3, gamma='auto', probability=True))
        self.model.fit(Train_X_Tfidf, Train_Y)
        # predict the labels on validation dataset
        #predictions_SVM = self.model.predict(Test_X_Tfidf)
        # Use accuracy_score function to get the accuracy
        #print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
        
        return self.model

    def predict(self, df):
        predictions = self.model.predict(df)
        confidenceList = np.amax(self.model.predict_proba(df), axis=1)
        
        return predictions, confidenceList