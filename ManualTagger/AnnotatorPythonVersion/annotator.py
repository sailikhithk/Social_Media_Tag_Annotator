'''
@file       Annotator.py
@date       2020/08/25
@brief      Top level class that defines the annotation tool and active learning algorithm
'''
from TagPredictor.multilabelclassifier_SVM import MultilabelClassifier_SVM
from TagPredictor.TagPredictor import TagPredictor
from ClassHumanTagger import ManualTagger
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import ast

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score

# Set Pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)   # Disable wrapping


'''
@brief  NLP classification annotation tool
'''
class Annotator:
    groundTruthDB = None            # Pandas dataframe of all data with ground truth labels
    labeledDB = None                # Pandas dataframe of labeled data
    unlabeledDB = None              # Pandas dataframe of unlabeled data

    tagPredictor = None             # TagPredictor object
    manualTagger = None             # ManualTagger object

    confidenceThreshold = 0.95      # Prediction confidence threshold to determine if a topic should be passed to ManualTagger


    '''
    @brief      Class constructor
    @param      datafile    CSV dataset file that contains topic text and tags
    @return     None
    '''
    def __init__(self, datafile):
        # Create databases
        self.groundTruthDB, self.labeledDB, self.unlabeledDB = self.createDatabases(datafile)

        # Set up ManualTagger
        self.manualTagger = ManualTagger()
    

    '''
    @brief      Performs preprocessing and cleaning on a sentence
    @param      text    String that contains the raw sentence
    @return     text    String that contains the cleaned sentence
    '''
    def cleanText(self, text):
        # Function that checks if all characters in a string are ASCII
        def is_ascii(s):
            return all(ord(c) < 128 for c in s)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)

        # Replace newline and tab characters with spaces
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')

        # Convert all letters to lowercase
        text = text.lower()

        # Split feature string into a list to perform processing on each word
        wordList = text.split()

        # Remove all stop words
        #stop_words = set(stopwords.words('english'))
        #wordList = [word for word in wordList if not word in stop_words]

        # Remove all words to contain non-ASCII characters
        wordList = [word for word in wordList if is_ascii(word)]

        # Remove all leading/training punctuation, except for '$'
        punctuation = '!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        wordList = [word.strip(punctuation) for word in wordList]

        # Reconstruct text
        text = ' '.join(wordList)

        return text


    '''
    @brief      Loads data from CSV files into Pandas dataframes and performs cleanText() on all columns
    @param      datafile        CSV file with all data
    @return     groundTruthDB   Pandas dataframe of all data with ground truth labels
    @return     labeledDB       Pandas dataframe of the labeled data
    @return     unlabeledDB     Pandas dataframe of the unlabeled data
    '''
    def createDatabases(self, datafile):
        # Load CSV file as ground truth database
        groundTruthDB = pd.read_csv(datafile)

        # Combine topic title and leading comment columns
        groundTruthDB['Bag_of_Words'] = groundTruthDB['Topic Title'] + groundTruthDB['Leading Comment']
        groundTruthDB['Bag_of_Words'] = groundTruthDB['Bag_of_Words'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')
        
        # Apply cleanText() to the bag of words
        groundTruthDB['Bag_of_Words'] = groundTruthDB['Bag_of_Words'].apply(lambda x: self.cleanText(x))

        # Code to duplicate multi-tag topics if necessary
        '''
        #create an offset value
        offset = 0
        #the total number of unique comments
        total = len(groundTruthDB)
        for index, entry in enumerate(groundTruthDB['Bag_of_Words']):
            #create a duplicate if post has multiple tags
            tag_list = ast.literal_eval(groundTruthDB.loc[index, 'Tags'])
            text = groundTruthDB.loc[index,'Bag_of_Words']
            while (isinstance(tag_list, list) and len(tag_list) > 1):
                #print(index)
                #sets the tag for the duplicate to a string
                groundTruthDB.loc[total+offset, 'Tags'] = tag_list.pop()
                #Adds the duplicate to the end of the pandas dataframe
                groundTruthDB.loc[total+offset, 'Bag_of_Words'] = text
                offset = offset + 1
            #Changes the first tag to a string
            if (len(tag_list) == 1):
                groundTruthDB.loc[index, 'Tags'] = tag_list.pop()
            #Changes empty tags from lists to strings
            if (isinstance(groundTruthDB.loc[index, 'Tags'], list)):
                groundTruthDB.loc[index, 'Tags'] = ''
                # Not sure why this element is stored as '[]' instead of ''
        '''

        # Filter out topics with no tags
        groundTruthDB = groundTruthDB[groundTruthDB['Tags'].map(len) > 2]

        # Convert Tag column elements from strings to lists
        groundTruthDB['Tags'] = groundTruthDB.Tags.apply(lambda x: x[1:-1].split(','))

        # Take only a subset of the full dataset, if necessary
        #groundTruthDB = groundTruthDB.sample(1000)

        # Split ground truth database into labeled and unlabelled databases
        unlabeledDB, labeledDB = train_test_split(groundTruthDB, test_size=0.2)

        return groundTruthDB, labeledDB, unlabeledDB


    '''
    @brief      Demonstration function to run the entire annotator application
    @param      classifier      Scikit-learn like classifier class to be used
    @return     None
    '''
    def runApplication(self, classifier):
        # Create multilabel binarizer for metric calculations
        mlb = MultiLabelBinarizer()

        # Set up TagPredictor object
        tagPredictor = TagPredictor(classifier, self.labeledDB)

        # Train tagPredictor
        tagPredictor.train()

        # Predict tags for all unlabeled topics
        tagList, confidenceList = tagPredictor.predict(self.unlabeledDB['Bag_of_Words'])

        # Continue running the active learning loop as long as there are still low-confidence topics
        counter = 1
        print('Minimum Confidence: ', min(confidenceList))
        print('Maximum Confidence: ', max(confidenceList))
        while (any(p < self.confidenceThreshold for p in confidenceList) == True):
            # Print out active learning statistics
            print('Active Learning Iteration ', counter)
            print('Labeled Database Size: ', len(self.labeledDB))
            print('Unlabeled Database Size: ', len(self.unlabeledDB))
            trueLabelIndicatorMatrix = mlb.fit_transform(self.unlabeledDB['Tags'])
            predictedLabelIndicatorMatrix = mlb.transform(tagList)
            print('Hamming Loss: ', hamming_loss(trueLabelIndicatorMatrix, predictedLabelIndicatorMatrix))
            print('Accuracy: ', accuracy_score(trueLabelIndicatorMatrix, predictedLabelIndicatorMatrix))
            
            # Get low-confidence topic indices
            lowConfIndices = [i for i in range(len(confidenceList)) if confidenceList[i] < self.confidenceThreshold]

            # Pass low-confidence topics to the manual tagger
            lowConfTopics = self.unlabeledDB.iloc[lowConfIndices]
            labeledTopics = self.manualTagger.run(lowConfTopics)
            print("Going on")
            # Add manually tagged topics to the labeled database
            self.labeledDB = pd.concat([self.labeledDB, labeledTopics], join='inner')

            # Remove tagged topics from unlabeled database
            #old = len(self.unlabeledDB)
            cond = self.unlabeledDB['Bag_of_Words'].isin(lowConfTopics['Bag_of_Words'])
            self.unlabeledDB.drop(self.unlabeledDB[cond].index, inplace=True)
            #print(old - len(self.unlabeledDB))
            #print(len(lowConfTopics))

            # Exit active learning loop if there are no more topics in the unlabeled database
            if (len(self.unlabeledDB) == 0):
                break

            # Train tagPredictor with updated labeled database
            tagPredictor = TagPredictor(classifier, self.labeledDB)
            tagPredictor.train()

            # Predict tags for all unlabeled topics
            tagList, confidenceList = tagPredictor.predict(self.unlabeledDB['Bag_of_Words'])

            counter += 1



if __name__ == '__main__':
    # Path to CSV datafile
    datafile = 'StackOverflow_new_tags.csv'

    # Instantiate Annotator object
    annotator = Annotator(datafile)

    # Run annotation application
    annotator.runApplication(MultilabelClassifier_SVM)

