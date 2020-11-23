import streamlit as st
import pandas as pd
import numpy as np
from openpyxl import Workbook, load_workbook
import SessionState
import time

class ManualTagger:
    df_posts = None
    data_dic = None
    iterate_key = None
    curr_key = None
    session = None
    df_tags = None
    tags_list = None
    df_labeled = None
    
    def __init__(self):
        self.df_posts

        self.data_dic
        self.iterate_key
        self.curr_key
        #self.session
        self.df_tags = pd.read_csv("Tags.csv") #Gets the list of possible tags
        self.tags_list = self.df_tags.Tags.tolist()
        self.df_labeled = pd.read_csv('LabeledData.csv') #Reads in the initial empty labeled data files
    """
    This function takes in a dataframe and returns a dictionary with the topic title as key and the leading comment as value
    """
    #@st.cache(suppress_st_warning=True, allow_output_mutation=True) # This function will be cashed and won't be run again
    def load_data(self, unlabeled_data):

        def get_data_from_file(df):
            dic = {}
            for row in range(len(df)):
                dic[df['Topic Title'].iloc[row]] = df['Leading Comment'].iloc[row]
            return dic
            
        #data_dic: contains the the scrapped data: with the topic title as key and the leading comment as value
        self.data_dic = get_data_from_file(self.df_posts)
        #iterate_key: an iterater that goes to the next topic when called next(iterate_key)
        self.iterate_key = iter(self.data_dic.keys())
        #curr_key: the tracker that keeps track of the current topic title that we are on
        self.curr_key = next(self.iterate_key)

    def run(self, dataframe):
        self.df_posts = dataframe
        if self.df_posts.empty:
            st.write("""
                        ** ML July Team1 Manual Tagging App**
                        All taggings are complete! Thank you for your help!
                    """)
            return self.df_labeled
        else:
            #Below is mainly code for StreamLit display.
            st.write("""
            ** ML July Team1 Manual Tagging App**
            """)
            self.load_data(self.df_posts)
            #remove the tagged post from the dataset and reset
            
            st.write("**_Topic Title:_**")
            st.write(self.curr_key)

            st.write("**_Leading Comment:_**")
            st.write(self.data_dic[self.curr_key])
            
            self.session = SessionState.get(run_id=0)
            options = st.multiselect('Please select suitable tags for the above topic.', self.tags_list, key=self.session.run_id)
            st.write('**_You have selected:_**', options)

            #writes the tagged post to a excel file and removes the tagged post from the dataset and reset
            st.write("""
            ** IMPORTANT: Once you hit "Submit" the label is perminant. Please make sure your decision is final before hitting submit! You have 2 minutes to complete the current topic tagging**
            """)
            if st.button("Submit"):
                st.write("**You have submitted your choices**")
                row_to_append = pd.DataFrame({self.df_labeled.columns[0]: [self.curr_key], self.df_labeled.columns[1]: [self.data_dic[self.curr_key]], self.df_labeled.columns[2]: [options]})
                self.df_labeled = self.df_labeled.append(row_to_append)
                self.df_labeled.to_csv('LabeledData.csv', index=False)
                df_posts = pd.read_csv("StackOverflow_new_tags.csv")
                df_posts = df_posts.iloc[1:]
                df_posts.to_csv("StackOverflow_new_tags.csv")
                
            #displays next topic
            self.session = SessionState.get(run_id=0)
            if st.button("Next Topic"):
                st.write("**Going to the next topic**")
                self.session.run_id += 1
                
            time.sleep(10)
            st.write("""
            ** Sleep Time Over! **
            """)


