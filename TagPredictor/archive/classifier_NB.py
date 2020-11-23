from sklearn import naive_bayes
from sklearn.metrics import accuracy_score

from classifier import Classifier

class Classifier_NB(Classifier):
    def run(self, Train_X_Tfidf, Train_Y, Test_X_Tfidf, Test_Y):
        print("Running Naive Bayes Classifier")
        
        # fit the training dataset on the NB classifier
        Naive = naive_bayes.MultinomialNB()
        Naive.fit(Train_X_Tfidf,Train_Y)
        # predict the labels on validation dataset
        predictions_NB = Naive.predict(Test_X_Tfidf)
        # Use accuracy_score function to get the accuracy
        print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)