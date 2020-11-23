import streamlit as st
import pandas as pd
import numpy as np


"""
This function takes in a dataframe and returns a dictionary with the topic title as key and the leading comment as value
"""
def get_data_from_file(df):
	dic = {}
	for row in df.itertuples():
		dic[row._2] = row._4
	return dic

"""
These two statements read in our desired csv files. The first one reads in the file that contains all redefined TagNames, which we have to make ourselves. The second contains scrapped data.
"""
#df_all_tag = pd.read_csv("FileWithAllTagName.csv")
df_topics = pd.read_csv("StackOverflow_new_tags.csv")


#new_labeled_topics: is the dataframe where we store labeled data
new_labeled_topics = df_topics.copy()
#data_dic: contains the the scrapped data: with the topic title as key and the leading comment as value
data_dic = get_data_from_file(df_topics)
#iterate_key: an iterater that goes to the next topic when called next(iterate_key)
iterate_key = iter(data_dic.keys())
#curr_key: the tracker that keeps track of the current topic title that we are on
curr_key = next(iterate_key)


"""
Below is mainly code for StreamLit display.
"""
st.write("""
** ML July Team1 Manual Tagging App**
""")

#the line below is for the drop down menu for tag selection. We will switch df_topics with df_all_tag.
options = st.multiselect('Please select suitable tags for the following topic.', df_topics['Tags'].unique())
st.write('You selected:', options)

#This construct the button and sets its function
if st.button("Next Topic"):
	curr_key = next(iterate_key)

st.write("""
	**Topic Title**
	""")
st.write(curr_key)

st.write("""
	**Leading Comment**
	""")
st.write(data_dic[curr_key])

#This intends to store the user input into the new_labeled_topics dataframe, but is not working yet
topi.new_labeled_topics.ix[new_labeled_topics._2 == topic.curr_key, 'Tags'] = options
