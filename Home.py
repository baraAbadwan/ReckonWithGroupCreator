import pandas as pd
import streamlit as st
import string
import itertools

def match_availability_and_allocate(group, group_size=3, id_col='A', freq_col='H', day_col='I', time_col='J'):
    """
    This function prepares the regrouping map and strict criteria, then allocates participants
    to groups based on the modular approach using flexibility ranking and balancing strategies.
    
    Parameters:
    - group (pd.DataFrame): A subgroup of participants to be processed.
    - group_size (int): The target size for each group (default is 3).
    
    Returns:
    - final_groups (list of pd.DataFrame): A list of DataFrames, each representing a final group.
    """
    # Prepare the regrouping map and strict criteria
    regrouping_map = {
        'I’m open to either': ['Weekly (60 minute sessions)', 'Bi-weekly (90 minute sessions)'],
        'Weekday, Weekend': ['Weekday', 'Weekend'],
        'Morning, Afternoon': ['Morning', 'Afternoon'],
        'Afternoon, Evening': ['Afternoon', 'Evening'],
        'Morning, Evening': ['Morning', 'Evening'],
        'Morning, Afternoon, Evening': ['Morning', 'Afternoon', 'Evening']
    }
    
    strict_frequencies = ['Weekly (60 minute sessions)', 'Bi-weekly (90 minute sessions)']
    strict_days = ['Weekday', 'Weekend']
    strict_times = ['Morning', 'Afternoon', 'Evening']

    flexibility_map = {
        freq_col: {'I’m open to either': 2, 'Weekly (60 minute sessions)': 1, 'Bi-weekly (90 minute sessions)': 1},
        day_col: {'Weekday, Weekend': 2, 'Weekday': 1, 'Weekend': 1},
        time_col: {'Morning, Afternoon, Evening': 3, 'Morning, Afternoon': 2, 'Afternoon, Evening': 2,
              'Morning, Evening': 2, 'Morning': 1, 'Afternoon': 1, 'Evening': 1}
    }


    # Allocate participants using the new modular approach
    final_groups = allocate_participants(
        group, 
        regrouping_map, 
        flexibility_map,
        strict_frequencies, 
        strict_days, 
        strict_times, 
        group_size=group_size,
        id_col=id_col,
        freq_col=freq_col,
        day_col=day_col,
        time_col=time_col
    )
    
    return final_groups

# Step 0: Rank participants by flexibility
def rank_participants_by_flexibility(df, flexibility_map, freq_col='H', day_col='I', time_col='J'):
    
    # Calculate flexibility score for each participant
    df['flexibility_score'] = (
        df[freq_col].map(flexibility_map[freq_col]) +
        df[day_col].map(flexibility_map[day_col]) +
        df[time_col].map(flexibility_map[time_col])
    )
    
    # Sort participants by flexibility score (ascending order, less flexible first)
    df = df.sort_values(by='flexibility_score', ascending=True)
    
    return df

# Step 1: Assign strict participants to groups
def assign_strict_participants(df, strict_frequencies, strict_days, strict_times, freq_col='H', day_col='I', time_col='J'):
    # Initialize strict groups as empty DataFrames for each combination
    #  Generate all possible strict combinations

    strict_combinations = pd.DataFrame(itertools.product(strict_frequencies, strict_days, strict_times))

    # strict_combinations = df[['H', 'I', 'J']].drop_duplicates()
    strict_groups = {tuple(row): pd.DataFrame() for row in strict_combinations.to_records(index=False)}

    # Assign strict participants to their respective groups
    strict_participants = df[
        (df[freq_col].isin(strict_frequencies)) & 
        (df[day_col].isin(strict_days)) & 
        (df[time_col].isin(strict_times))
    ]

    for _, participant in strict_participants.iterrows():
        group_key = (participant[freq_col], participant[day_col], participant[time_col])
        strict_groups[group_key] = pd.concat([strict_groups[group_key], participant.to_frame().T], ignore_index=True)

    return strict_groups, strict_participants

# Step 2: Analyze group sizes
def analyze_group_sizes(groups):
    group_sizes = {group_key: len(group_df) for group_key, group_df in groups.items()}
    return group_sizes

# Step 3: Assign less flexible participants to groups
def assign_less_flexible_participants(flexible_participants, groups, group_sizes, regrouping_map, group_size=3, freq_col='H', day_col='I', time_col='J'):
    for _, participant in flexible_participants.iterrows():
        # Get participant's criteria
        freq = participant[freq_col]
        day = participant[day_col]
        time = participant[time_col]
        # Find all strict possibilities for this participant using regrouping_map
        possible_freqs = regrouping_map.get(freq, [freq])
        possible_days = regrouping_map.get(day, [day])
        possible_times = regrouping_map.get(time, [time])


        # Track the best group to assign the participant to
        best_group = None
        min_mod_difference = float('inf')  # Minimum difference from being divisible by 3
        largest_group_size = -1  # Track largest group size

        for strict_combination in itertools.product(possible_freqs, possible_days, possible_times):
            if strict_combination in groups:
    
                current_size = group_sizes[strict_combination]
                mod_difference = (group_size - (current_size+1 % group_size)) % group_size  # How far from being divisible by 3
                
                # Priority 1: Groups closest to being divisible by 3
                if mod_difference < min_mod_difference:
                    best_group = strict_combination
                    min_mod_difference = mod_difference
                    largest_group_size = current_size  # Reset largest group size to this group
                
                # Priority 2: If multiple groups have the same mod_difference, choose the largest one
                elif mod_difference == min_mod_difference and current_size > largest_group_size:
                    best_group = strict_combination
                    largest_group_size = current_size
        
        # Assign the participant to the best group found
        if best_group is not None:
            groups[best_group] = pd.concat([groups[best_group], participant.to_frame().T], ignore_index=True)
            group_sizes[best_group] += 1
        else:
            # Priority 3: Create a new group if necessary
            new_group_key = (freq, day, time)
            groups[new_group_key] = participant.to_frame().T
            group_sizes[new_group_key] = 1
    
    return groups, group_sizes


def allocate_participants(df, regrouping_map, flexibility_map, strict_frequencies, strict_days, strict_times, group_size=3, id_col = 'A', freq_col='H', day_col='I', time_col='J' ):
    # Step 0: Rank participants by flexibility
    df = rank_participants_by_flexibility(df, flexibility_map, freq_col, day_col, time_col)
    
    # Step 1: Assign strict participants to groups
    strict_groups, strict_participants = assign_strict_participants(df, strict_frequencies, strict_days, strict_times, freq_col, day_col, time_col)
    
    # Step 2: Analyze group sizes
    group_sizes = analyze_group_sizes(strict_groups)
    
    # Step 3: Assign less flexible participants to groups
    less_flexible_participants = df.loc[~df[id_col].isin(strict_participants[id_col])]
    strict_groups, group_sizes = assign_less_flexible_participants(less_flexible_participants, strict_groups, group_sizes, regrouping_map, group_size)
    
    # Filter out empty groups
    final_groups = {key: group for key, group in strict_groups.items() if not group.empty}
    
    # Collect unassigned participants based on participant identifier in column 'A'
    all_assigned_participants = pd.concat(final_groups.values())[id_col]
    unassigned_participants = df.loc[~df[id_col].isin(all_assigned_participants)]
    
    if not unassigned_participants.empty:
        final_groups['Unassigned'] = unassigned_participants
    
    # Convert final groups into a list of DataFrames
    group_dfs = list(final_groups.values())
    
    return group_dfs




def timezone_fixer(df, current_timezone_col = 'E', future_timezone_col='G'):
    df[current_timezone_col] = df.apply(lambda x: x[future_timezone_col].split('(')[1].split(')')[0] if type(x[future_timezone_col]) is str else x[current_timezone_col], axis =1)
    timezone_mapping = {
        'EST': 'EST',
        'Eastern Standard Time': 'EST',
        'CST': 'CST',
        'Central Standard Time': 'CST',
        'Mountain Standard Time': 'MST',
        'Pacific Standard Time': 'PST',
        'Alaska Standard Time': 'AKST'
        }
    df[current_timezone_col] = df[current_timezone_col].apply(lambda x : timezone_mapping[x] if x in timezone_mapping.keys() else x)
    return df
    

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
    column_names = dict(zip(column_labels, df.columns))

    df = df.rename(columns=column_mapping)

    df = timezone_fixer(df)
    df = df.loc[df['D']=='I would like Reckon With to put me in a group based on my survey results']
    return df, column_names

def filter_by_racial_literacy(df, on_col='AO'):
    df_list = condition_split(df, on_col, [(1, 2, 3), (4, 5), (6,7)])
    return df_list

def condition_split(df, on_col, conditions):
    df_list = [df[df[on_col].isin(condition)] for condition in conditions]
    return df_list

def hard_split(df, on_col):
    # Get unique values in the specified column
    unique_values = df[on_col].unique()
    # Create a separate DataFrame for each unique value
    df_list = [df.loc[df[on_col] == value].copy() for value in unique_values]
    
    return df_list

def split_into_groups(group, n):
    # Shuffle the DataFrame
    shuffled_group = group.sample(frac=1).reset_index(drop=True)
    
    # Split into chunks of size n
    subgroups = [shuffled_group.iloc[i:i+n] for i in range(0, len(shuffled_group), n)]
    
    return subgroups


# Main Streamlit app
st.title("Participant Grouping Based on Multiple Criteria")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df, column_names = load_prep_data(uploaded_file)
    # st.write(column_names['AC'])

    # User input for grouping criteria
    group_by = st.multiselect(
        "Select criteria for grouping",
        options=['Racial Literacy', 'Timezone', 'Matching Availability', 'Into Threes']
    )


    # Initialize the groups list with the entire dataframe as the starting point
    groups = [df]
    group_labels = ['All Participants']

    # Racial Literacy split
    if 'Racial Literacy' in group_by:
        racial_literacy_groups = filter_by_racial_literacy(df)
        racial_literacy_labels = ["Racial Literacy Low", "Racial Literacy Medium", "Racial Literacy High"]

        groups = racial_literacy_groups
        group_labels = racial_literacy_labels

    # Timezone split
    if 'Timezone' in group_by:
        new_groups = []
        new_labels = []

        for group, label in zip(groups, group_labels):
            timezone_splits = hard_split(group, 'E')
            for subgroup in timezone_splits:
                # Append the new groups and labels
                new_groups.append(subgroup)
                timezone_label = f"Timezone {subgroup['E'].iloc[0]}"
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

    # Display the groups
    for group, label in zip(groups, group_labels):
        st.write(label)
        st.write(group)