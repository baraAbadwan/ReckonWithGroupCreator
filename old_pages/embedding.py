import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import string
import numpy as np

# Function to generate column labels
def generate_column_letters(num_columns):
    letters = list(string.ascii_uppercase)
    column_labels = []
    
    # Create single-letter labels (A-Z)
    column_labels.extend(letters)
    
    # Create two-letter labels (AA, AB, ..., ZZ)
    for first_letter in letters:
        for second_letter in letters:
            column_labels.append(first_letter + second_letter)
            if len(column_labels) >= num_columns:
                return column_labels

# Function to load and prepare the data
def load_prep_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    column_labels = generate_column_letters(len(df.columns))
    column_mapping = dict(zip(df.columns, column_labels))
    df = df.rename(columns=column_mapping)
    return df

# Function to compute sentence embeddings
def compute_embeddings(df, model, columns_to_embed):
    # Join all answers from specified columns into a single string per participant
    sentences = df[columns_to_embed].fillna('').agg(' '.join, axis=1).tolist()
    embeddings = model.encode(sentences)
    return embeddings

# Function to group participants using nearest neighbors
def nearest_neighbors_grouping(embeddings, group_size=3):
    nn = NearestNeighbors(n_neighbors=group_size, algorithm='auto').fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    
    groups = []
    for idx, group in enumerate(indices):
        groups.append([f"Participant {i+1}" for i in group])
    
    grouped_df = pd.DataFrame(groups)
    grouped_df.index = [f'Group {i+1}' for i in range(len(groups))]
    return grouped_df

# Step 1: Upload CSV and load data
# st.title("Non-Profit Group Matching with Similarity")
# uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
uploaded_file='data/participant_data.csv'

if uploaded_file:
    # Step 2: Load the CSV into a pandas dataframe
    df = load_prep_data(uploaded_file)
    
    # Step 3: Filter based on Column D (keep only those who want to be matched by Reckon With)
    filtered_df = df[df['D'] == 'I would like Reckon With to put me in a group based on my survey results']
    
    st.write("Filtered Data Based on Column D:")
    st.write(filtered_df)
    
    # Step 4: Compute embeddings for the selected columns
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose any other pre-trained model
    columns_to_embed = filtered_df.columns[1:]  # Exclude the Participant ID column (Column A)
    
    embeddings = compute_embeddings(filtered_df, model, columns_to_embed)
    
    # Step 5: Perform nearest neighbors grouping
    # group_size = st.number_input("Select group size", min_value=2, max_value=10, value=3)
    group_size = 3
    grouped_df = nearest_neighbors_grouping(embeddings, group_size)
    
    grouped_df.to_csv('data/grouping_test.csv')
    print(grouped_df)
    # # Step 6: Display and download the grouped data
    # st.write("Grouped Participants Based on Similarity:")
    # st.dataframe(grouped_df)
    
    # st.download_button(
    #     label="Download Grouped Data",
    #     data=grouped_df.to_csv(index=False),
    #     file_name='grouped_participants.csv',
    #     mime='text/csv',
    # )