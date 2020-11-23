import numpy as np

from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

from classifier import Classifier

class MultilabelClassifier_SVM(Classifier):
    kernel = None
    model = None

    def __init__(self, kernel='linear'):
        self.kernel = kernel


    def train(self, Train_X_Tfidf, Train_Y, Test_X_Tfidf=None, Test_Y=None):
        # Fit the training dataset on the classifier
        self.model = MultiOutputClassifier(SVC(C=1.0, kernel=self.kernel, degree=3, gamma='auto', probability=True))
        self.model.fit(Train_X_Tfidf, Train_Y)
        
        return self.model


    def predict(self, df):
        # Raw list of category probabilities (Shape n_categories x n_samples x n_outputs)
        rawList = self.model.predict_proba(df)
        
        # Convert to NumPy array
        # Extract relevant output
        # Transpose and round probabilities to create an indicator matrix
        predictionMatrix = np.round_(np.array(rawList)[:, :, 1]).T
        
        # Extract relevant probability output
        # Average confidences across all categories for all samples
        probabilityMatrix = np.amax(np.array(rawList), axis=2)
        confidenceList = np.average(probabilityMatrix.T, axis=1)
        
        return predictionMatrix, confidenceList