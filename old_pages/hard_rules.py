import pandas as pd
import streamlit as st
import string
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
            
def load_prep_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    column_labels = generate_column_letters(len(df.columns))
    column_mapping = dict(zip(df.columns, column_labels))
    df = df.rename(columns=column_mapping)
    return df
# Function to filter participants by multiple criteria
def filter_participants(df):
    # Frequency filter
    weekly = df[df['H'].apply(lambda x: x in ['Weekly (60 minute sessions)', 'I’m open to either'])]
    biweekly = df[df['H'].apply(lambda x: x in ['Bi-weekly (90 minute sessions)', 'I’m open to either'])]
    # open_to_either = df[df['H'] == 'I’m open to either']
    
    # Time preferences (Weekday/Weekend)
    weekday = df[df['I'].str.contains('Weekday')]
    weekend = df[df['I'].str.contains('Weekend')]
    
    # Time of day (Morning/Afternoon/Evening)
    morning = df[df['J'].str.contains('Morning')]
    afternoon = df[df['J'].str.contains('Afternoon')]
    evening = df[df['J'].str.contains('Evening')]
    
    # Racial literacy
    group_1_3 = df[df['AO'].isin([1, 2, 3])]
    group_4_5 = df[df['AO'].isin([4, 5])]
    group_6_7 = df[df['AO'].isin([6, 7])]
    
    return weekly, biweekly, weekday, weekend, morning, afternoon, evening, group_1_3, group_4_5, group_6_7

# Function to match participants across multiple criteria
def match_participants(df):
    # Get filtered groups based on criteria
    weekly, biweekly, weekday, weekend, morning, afternoon, evening, group_1_3, group_4_5, group_6_7 = filter_participants(df)
    
    st.write(group_1_3)
    st.write(group_4_5)
    st.write(group_6_7)

    # Example: Matching weekly participants with weekday and morning preferences who fall into the racial literacy group 1-3
    group = weekly[ weekly.index.isin(group_1_3.index)]
    st.write(weekly)
    # Extend the logic to other combinations
    # Combine criteria to form final groups
    final_groups = []
    for participant in group.index:
        final_groups.append(df.loc[participant])
    
    return pd.DataFrame(final_groups)

# Main Streamlit app
st.title("Participant Grouping Based on Multiple Criteria")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
uploaded_file='data/participant_data.csv'
if uploaded_file:
    df = load_prep_data(uploaded_file)
    df = df.set_index('D').loc['I would like Reckon With to put me in a group based on my survey results'].reset_index()
    # Match participants across all criteria
    grouped_df = match_participants(df)
    
    st.write("Participants grouped based on multiple criteria:")
    st.dataframe(grouped_df)
    
    st.download_button(
        label="Download Grouped Data",
        data=grouped_df.to_csv(index=False),
        file_name='grouped_participants.csv',
        mime='text/csv',
    )
