from abc import ABC, abstractmethod

class Classifier(ABC):
    @abstractmethod
    def train(self, Train_X_Tfidf, Train_Y, Test_X_Tfidf, Test_Y):
        pass

    @abstractmethod
    def predict(self, df):
        pass