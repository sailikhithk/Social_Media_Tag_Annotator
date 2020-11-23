import pandas as pd

#User Interface
#Save low confidence data into a file for later analysis
#Print outs the low confidence data
class ManualTagger:
    groundTruthDB = None
    
    def __init__(self, data_frame):
        self.groundTruthDB = data_frame
        
        
    def run(self, topicsDF):
        #filt = (self.groundTruthDF['Bag_of_Words'] == bagOfWords)
        #tag_in_df = self.groundTruthDF.loc[filt, 'Tags'] #filt gives us the row we want, and 'Tag' gives us the column that we want
        cond = self.groundTruthDB['Bag_of_Words'].isin(topicsDF['Bag_of_Words'])
        labeledTopicsDF = self.groundTruthDB[cond]
        return  labeledTopicsDF