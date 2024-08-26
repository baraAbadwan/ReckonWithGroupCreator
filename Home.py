import pandas as pd
import streamlit as st
from src.loader import load_prep_data
from src.grouping import match_availability_and_allocate, filter_by_racial_literacy, filter_by_family_arrival, map_to_closest_timezone, hard_split, split_into_groups

def prepare_groups_for_download(groups, group_labels, column_names):
    # Add a 'Group' column to each group and concatenate them into a single DataFrame
    labeled_groups = []
    for group, label in zip(groups, group_labels):
        group_with_label = group.copy()
        group_with_label['Group'] = label
        labeled_groups.append(group_with_label)
    
    # Concatenate all groups into one DataFrame
    combined_df = pd.concat(labeled_groups, ignore_index=True)
    
    # Move 'Group' column to be the first column
    cols = ['Group'] + [col for col in combined_df.columns if col != 'Group']
    combined_df = combined_df[cols]
    
    # Sort the DataFrame by the 'Group' column
    combined_df = combined_df.sort_values(by='Group')
    combined_df = combined_df.rename(column_names, axis=1)
    
    return combined_df




# Main Streamlit app
st.title("Participant Grouping Based on Multiple Criteria")
col1, col2 = st.columns([1, 4])

with col2:
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")


if uploaded_file:
    df, column_names = load_prep_data(uploaded_file)
    # st.write(column_names['AC'])

    # User input for grouping criteria
    with col2:
        group_by = st.multiselect(
            "Select criteria for grouping",
            options=['Racial Literacy', 
                     'Timezone', 
                     'Matching Availability', 
                     'Into Threes', 
                     'Family Arrival', 
                     'Class', 
                     'Geography', 
                     'Age', 
                     'Religion', 
                     'Caregiving',
                     'Childhood class', 
                     'Childhood Geography']
        )


    # Initialize the groups list with the entire dataframe as the starting point
    groups = [df]
    group_labels = ['All Participants']


    if 'Family Arrival' in group_by:
        default_bins = [('1600s', '1700s'), ('1700s', '1800s'), ('1900s', '2000s')]

        # Possible scores for racial literacy
        possible_scores = list(['1600s', '1700s', '1800s', '1900s','2000s'])  # Assuming scores are in the range of 1 to 7

        # Streamlit multiselect inputs to allow users to modify the bins
        with col1:
            low_bin = st.multiselect("Select scores for Arrival 1", possible_scores, default_bins[0])
            medium_bin = st.multiselect("Select scores for Arrival 2", possible_scores, default_bins[1])
            high_bin = st.multiselect("Select scores for Arrival 3", possible_scores, default_bins[2])

        # Create the bins list from user input
        bins = [low_bin, medium_bin, high_bin]

        # Process each group individually
        new_groups = []
        new_labels = []

        for group, label in zip(groups, group_labels):
            racial_literacy_splits = filter_by_family_arrival(group, bins)
            for subgroup, r_label in zip(racial_literacy_splits, ["Old", "Medium", "New"]):
                new_groups.append(subgroup)
                new_labels.append(f"{label}, Arrival {r_label}")

        groups = new_groups
        group_labels = new_labels


    if 'Racial Literacy' in group_by:
        default_bins = [(1, 2, 3), (4, 5), (6, 7)]

        # Possible scores for racial literacy
        possible_scores = list(range(1, 8))  # Assuming scores are in the range of 1 to 7

        # Streamlit multiselect inputs to allow users to modify the bins
        with col1:
            low_bin = st.multiselect("Select scores for Low Racial Literacy", possible_scores, default_bins[0])
            medium_bin = st.multiselect("Select scores for Medium Racial Literacy", possible_scores, default_bins[1])
            high_bin = st.multiselect("Select scores for High Racial Literacy", possible_scores, default_bins[2])

        # Create the bins list from user input
        bins = [low_bin, medium_bin, high_bin]

        # Process each group individually
        new_groups = []
        new_labels = []

        for group, label in zip(groups, group_labels):
            racial_literacy_splits = filter_by_racial_literacy(group, bins)
            for subgroup, r_label in zip(racial_literacy_splits, ["Low", "Medium", "High"]):
                new_groups.append(subgroup)
                new_labels.append(f"{label}, Racial Literacy {r_label}")

        groups = new_groups
        group_labels = new_labels

    # Timezone split
    if 'Timezone' in group_by:
        new_groups = []
        new_labels = []

        with col1:
            collate_timezones = st.checkbox('Map to closest major timezone?', True)
            if collate_timezones:
                groups = [map_to_closest_timezone(group) for group in groups]


        for group, label in zip(groups, group_labels):
            timezone_splits = hard_split(group, 'E')
            for subgroup in timezone_splits:
                # Append the new groups and labels
                new_groups.append(subgroup)
                timezone_label = f"Timezone {subgroup['E'].iloc[0]}"
                new_labels.append(f"{label}, {timezone_label}" if label != "All Participants" else timezone_label)

        groups = new_groups
        group_labels = new_labels

    if 'Class' in group_by:
        new_groups = []
        new_labels = []

        for group, label in zip(groups, group_labels):
            timezone_splits = hard_split(group, 'O')
            for subgroup in timezone_splits:
                # Append the new groups and labels
                new_groups.append(subgroup)
                timezone_label = f"Class {subgroup['O'].iloc[0]}"
                new_labels.append(f"{label}, {timezone_label}" if label != "All Participants" else timezone_label)

        groups = new_groups
        group_labels = new_labels


    if 'Childhood class' in group_by:
        new_groups = []
        new_labels = []

        for group, label in zip(groups, group_labels):
            timezone_splits = hard_split(group, 'N')
            for subgroup in timezone_splits:
                # Append the new groups and labels
                new_groups.append(subgroup)
                timezone_label = f"Childhood class {subgroup['N'].iloc[0]}"
                new_labels.append(f"{label}, {timezone_label}" if label != "All Participants" else timezone_label)

        groups = new_groups
        group_labels = new_labels

    if 'Geography' in group_by:
        new_groups = []
        new_labels = []

        for group, label in zip(groups, group_labels):
            timezone_splits = hard_split(group, 'M')
            for subgroup in timezone_splits:
                # Append the new groups and labels
                new_groups.append(subgroup)
                timezone_label = f"Geography {subgroup['M'].iloc[0]}"
                new_labels.append(f"{label}, {timezone_label}" if label != "All Participants" else timezone_label)

        groups = new_groups
        group_labels = new_labels
    
    if 'Childhood Geography' in group_by:
        new_groups = []
        new_labels = []

        for group, label in zip(groups, group_labels):
            timezone_splits = hard_split(group, 'L')
            for subgroup in timezone_splits:
                # Append the new groups and labels
                new_groups.append(subgroup)
                timezone_label = f"Childhood Geography {subgroup['L'].iloc[0]}"
                new_labels.append(f"{label}, {timezone_label}" if label != "All Participants" else timezone_label)

        groups = new_groups
        group_labels = new_labels

    if 'Age' in group_by:
        new_groups = []
        new_labels = []

        for group, label in zip(groups, group_labels):
            timezone_splits = hard_split(group, 'K')
            for subgroup in timezone_splits:
                # Append the new groups and labels
                new_groups.append(subgroup)
                timezone_label = f"Age {subgroup['K'].iloc[0]}"
                new_labels.append(f"{label}, {timezone_label}" if label != "All Participants" else timezone_label)

        groups = new_groups
        group_labels = new_labels


    if 'Religion' in group_by:
        new_groups = []
        new_labels = []

        for group, label in zip(groups, group_labels):
            group['R'] = group['R'].fillna('Other')
            group['R'] = group['R'].apply(lambda x: x.split(',')[0])

            timezone_splits = hard_split(group, 'R')
            for subgroup in timezone_splits:
                # Append the new groups and labels
                new_groups.append(subgroup)
                timezone_label = f"Religion {subgroup['R'].iloc[0]}"
                new_labels.append(f"{label}, {timezone_label}" if label != "All Participants" else timezone_label)

        groups = new_groups
        group_labels = new_labels

    if 'Caregiving' in group_by:
        new_groups = []
        new_labels = []

        for group, label in zip(groups, group_labels):

            timezone_splits = hard_split(group, 'Y')
            for subgroup in timezone_splits:
                # Append the new groups and labels
                new_groups.append(subgroup)
                timezone_label = f"Caregiving {subgroup['Y'].iloc[0]}"
                new_labels.append(f"{label}, {timezone_label}" if label != "All Participants" else timezone_label)

        groups = new_groups
        group_labels = new_labels

    # Matching Availability split
    if 'Matching Availability' in group_by:
        new_groups = []
        new_labels = []

        for group, label in zip(groups, group_labels):
            # Apply matching availability to each subgroup
            final_subgroups = match_availability_and_allocate(group, group_size=3)
            
            # Append the new subgroups and labels
            new_groups.extend(final_subgroups)
            new_labels.extend([f"{label}, Availability {i+1}" for i in range(len(final_subgroups))])
        
        groups = new_groups
        group_labels = new_labels



    # Into Threes split
    if 'Into Threes' in group_by:
        new_groups = []
        new_labels = []

        for group, label in zip(groups, group_labels):
            three_splits = split_into_groups(group, 3)
            for i, subgroup in enumerate(three_splits):
                new_groups.append(subgroup)
                new_labels.append(f"{label}, Group {i + 1}")

        groups = new_groups
        group_labels = new_labels

    
    group_labels = [label.replace("All Participants,", "") for label in group_labels]

    # Display the groups
    with col2:
        for group, label in zip(groups, group_labels):
            st.write(label)
            st.write(group)


    combined_df = prepare_groups_for_download(groups, group_labels, column_names)
    
    # Convert DataFrame to CSV
    csv = combined_df.to_csv(index=False).encode('utf-8')
    
    # Provide download button
    st.download_button(
        label="Download Groups as CSV",
        data=csv,
        file_name='grouped_participants.csv',
        mime='text/csv',)