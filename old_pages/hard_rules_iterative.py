import pandas as pd
import streamlit as st
import string 
def generate_column_letters(num_columns):
    letters = list(string.ascii_uppercase)
    column_labels = []
    column_labels.extend(letters)
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

# Function to filter participants by racial literacy groups
def filter_by_racial_literacy(df):
    group_1_3 = df[df['AO'].isin([1, 2, 3])]
    group_4_5 = df[df['AO'].isin([4, 5])]
    group_6_7 = df[df['AO'].isin([6, 7])]
    return group_1_3, group_4_5, group_6_7

# Function to group participants based on hard requirements
def form_groups(df):
    assigned_participants = set()
    final_groups = []

    # Define helper to form groups of three
    def form_group(filtered_df):
        group = []
        for _, participant in filtered_df.iterrows():
            if participant.name not in assigned_participants:
                group.append(participant.name)
                assigned_participants.add(participant.name)
            if len(group) == 3:  # Form a group of 3
                final_groups.append(group)
                group = []
        return group  # Return remaining participants that didn't form a full group
    
    # Handle participants with strict requirements
    def handle_strict_requirements(filtered_df):
        # Weekdays only
        weekdays_only = filtered_df[filtered_df['I'] == 'Weekday']
        remaining = form_group(weekdays_only)
        
        # Weekends only
        weekends_only = filtered_df[filtered_df['I'] == 'Weekend']
        remaining += form_group(weekends_only)
        
        # Combine remaining participants with flexible ones
        flexible = filtered_df[filtered_df['I'] == 'Weekday, Weekend']
        remaining += form_group(flexible)
        
        # Add any remaining participants to final groups
        if remaining:
            final_groups.append(remaining)
    
    handle_strict_requirements(df)
    return final_groups

# Main function to apply all steps
def match_participants(df):
    # Group participants by racial literacy
    group_1_3, group_4_5, group_6_7 = filter_by_racial_literacy(df)
    st.write()
    
    # Form groups for each racial literacy group
    final_groups = []
    final_groups += form_groups(group_1_3)
    final_groups += form_groups(group_4_5)
    final_groups += form_groups(group_6_7)
    
    # Convert the final groups to a DataFrame for easy display
    grouped_df = pd.DataFrame(final_groups).T  # Transpose for better viewing
    grouped_df.columns = [f'Group {i+1}' for i in range(len(final_groups))]
    
    return grouped_df

# Main Streamlit app
st.title("Participant Grouping Based on Multiple Criteria")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
uploaded_file='data/participant_data.csv'

if uploaded_file:
    df = load_prep_data(uploaded_file)
    df = df[df['D'] == 'I would like Reckon With to put me in a group based on my survey results'].reset_index(drop=True)
    
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