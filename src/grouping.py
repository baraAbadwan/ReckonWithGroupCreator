import pandas as pd
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

def assign_less_flexible_participants(flexible_participants, groups, group_sizes, regrouping_map, group_size=3, freq_col='H', day_col='I', time_col='J'):
    remaining_participants = flexible_participants.copy()  # Initialize remaining participants

    for _, participant in flexible_participants.iterrows():
        # Remove the current participant from remaining_participants
        remaining_participants = remaining_participants[remaining_participants['A'] != participant['A']]

        # Get participant's criteria
        freq = participant[freq_col]
        day = participant[day_col]
        time = participant[time_col]
        possible_freqs = regrouping_map.get(freq, [freq])
        possible_days = regrouping_map.get(day, [day])
        possible_times = regrouping_map.get(time, [time])

        # Track the best group to assign the participant to
        best_group = None
        min_mod_difference = float('inf')
        smallest_group_size = float('inf')

        for strict_combination in itertools.product(possible_freqs, possible_days, possible_times):
            if strict_combination in groups:
                current_size = group_sizes[strict_combination]
                mod_difference = (group_size - ((current_size + 1) % group_size)) % group_size
                
                # Calculate compatibility score using remaining participants
                compatibility_score = calculate_compatibility(strict_combination, remaining_participants, participant, regrouping_map,  freq_col, day_col, time_col)


                # Priority 1: Groups closest to being divisible by 3
                if mod_difference < min_mod_difference or (mod_difference == min_mod_difference and compatibility_score > smallest_group_size):
                    best_group = strict_combination
                    min_mod_difference = mod_difference
                    smallest_group_size = compatibility_score

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

def calculate_compatibility(group_combination, flexible_participants, current_participant,  regrouping_map,  freq_col='H', day_col='I', time_col='J'):
    # Placeholder function to calculate a compatibility score based on how many future participants
    # with similar flexibility could also fit in this group
    # For now, this could be a simple heuristic based on overlapping preferences
    score = 0
    for _, participant in flexible_participants.iterrows():
        if participant['A'] == current_participant['A']:
            continue
        # Check if this participant shares similar flexibility with the current participant
        if group_combination[0] in regrouping_map.get(participant[freq_col], [participant[freq_col]]) and \
           group_combination[1] in regrouping_map.get(participant[day_col], [participant[day_col]]) and \
           group_combination[2] in regrouping_map.get(participant[time_col], [participant[time_col]]):
            score += 1
    return score


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


def filter_by_racial_literacy(df, bins, on_col='AO'):
    # Split the DataFrame based on the bins provided
    df_list = condition_split(df, on_col, bins)
    
    return df_list

def filter_by_family_arrival(df, bins, on_col='AC'):
    # Split the DataFrame based on the bins provided
    df_list = condition_split(df, on_col, bins)
    
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

def map_to_closest_timezone(df, timezone_col='E'):
    # Define a mapping from less common time zones to their closest major time zones
    timezone_mapping = {
        'AKST': 'PST',  # Alaska Standard Time to Pacific Standard Time
        'MST': 'CST',   # Mountain Standard Time to Central Standard Time
        # Keep the major time zones unchanged
        'PST': 'PST',
        'CST': 'CST',
        'EST': 'EST'
    }
    
    # Apply the mapping to the DataFrame
    df[timezone_col] = df[timezone_col].map(timezone_mapping).fillna(df[timezone_col])
    
    return df