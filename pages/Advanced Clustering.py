import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap.umap_ as umap
import plotly.express as px
from src.loader import load_prep_data
import numpy as np

# Load your pre-trained language model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("Participant Grouping Based on Multiple Criteria")
col1, col2 = st.columns([1, 4])

with col2:
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df, column_names = load_prep_data(uploaded_file)
    st.write(df)
    
    subjective_response = st.multiselect(
        "Select subjective response columns",
        options=column_names.keys(),
        default=['AI']
    )
    
    # Combine subjective responses into one big string per participant
    df['combined_responses'] = df[subjective_response].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    st.write(df['combined_responses'])
    
    # Generate embeddings for each participant's combined responses
    embeddings = model.encode(df['combined_responses'].tolist())
    
    # Dimensionality reduction with UMAP
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine')
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Clustering to group participants
    n_clusters = len(df) // 3  # Adjust the number of clusters based on your requirement
    clustering_model = KMeans(n_clusters=n_clusters)
    labels = clustering_model.fit_predict(embeddings)
    
    # Add cluster labels and UMAP coordinates to the DataFrame
    df['Cluster'] = labels
    df['UMAP_1'] = embedding_2d[:, 0]
    df['UMAP_2'] = embedding_2d[:, 1]
    
    # Plot UMAP visualization using Plotly
    fig = px.scatter(df, x='UMAP_1', y='UMAP_2', color='Cluster',
                     title="UMAP projection of participant embeddings",
                     hover_data=['A', 'combined_responses'],
                     color_continuous_scale=px.colors.qualitative.Plotly)  # Use a qualitative color scale
    
    st.plotly_chart(fig)
    
    # Group participants by clusters
    grouped = df.groupby('Cluster').apply(lambda x: x[:3])  # Ensure groups of 3
    st.write(grouped)
