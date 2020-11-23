import streamlit as st
import pandas as pd
import numpy as np
from openpyxl import Workbook, load_workbook
import SessionState

"""
This function takes in a dataframe and returns a dictionary with the topic title as key and the leading comment as value
"""
#@st.cache(suppress_st_warning=True, allow_output_mutation=True) # This function will be cashed and won't be run again
def load_data(unlabeled_data):

    def get_data_from_file(df):
        dic = {}
        for row in range(len(df)):
            dic[df['Topic Title'].iloc[row]] = df['Leading Comment'].iloc[row]
        return dic

    """
    These two statements read in our desired csv files. The first one reads in the file that contains all redefined TagNames, which we have to make ourselves. The second contains scrapped data.
    """
    #df_all_tag = pd.read_csv("FileWithAllTagName.csv")
    df_posts = pd.read_csv(unlabeled_data)

    #new_labeled_topics: is the dataframe where we store labeled data
    new_labeled_topics = df_posts.copy()
    #data_dic: contains the the scrapped data: with the topic title as key and the leading comment as value
    data_dic = get_data_from_file(df_posts)
    #iterate_key: an iterater that goes to the next topic when called next(iterate_key)
    iterate_key = iter(data_dic.keys())
    #curr_key: the tracker that keeps track of the current topic title that we are on
    curr_key = next(iterate_key)

    return data_dic, iterate_key, curr_key   

"""
Below is mainly code for StreamLit display.
"""
st.write("""
** ML July Team1 Manual Tagging App**
""")

#the line below is for the drop down menu for tag selection. We will switch df_posts with df_all_tag.
data_dic, iterate_key, curr_key = load_data("StackOverflow_new_tags.csv")
#importing tags list
df_tags = pd.read_csv("Tags.csv")
tags_list = df_tags.Tags.tolist()

#remove the tagged post from the dataset and reset
if st.button("Reset"):
    df_posts = pd.read_csv("StackOverflow_new_tags.csv")
    df_posts = df_posts.iloc[1:]
    df_posts.to_csv("StackOverflow_new_tags.csv")

#displays next topic
session = SessionState.get(run_id=0)
if st.button("Next Topic"):
    session.run_id += 1

options = st.multiselect('Please select suitable tags for the following topic.', tags_list, key=session.run_id)
st.write('You selected:', options)

st.write("Topic Title:")
st.write(curr_key)

st.write("Leading Comment:")
st.write(data_dic[curr_key])

#writes the tagged post to a excel file
if st.button("Submit"):
    df = pd.read_csv('LabeledData.csv')
    row_to_append = pd.DataFrame({df.columns[0]: [curr_key], df.columns[1]: [data_dic[curr_key]], df.columns[2]: [options]})
    df = df.append(row_to_append)
    df.to_csv('LabeledData.csv', index=False)
