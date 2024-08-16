import streamlit as st 
import pandas as pd
import random
import itertools
def create_combinations():

    # Step 1: Dynamically generate all possible combinations of strict criteria
    meeting_frequencies = ['Weekly (60 minute sessions)', 'Bi-weekly (90 minute sessions)', 'I’m open to either'], 
    meeting_days = ['Weekday', 'Weekend', 'Weekday, Weekend']
    meeting_times = ['Morning', 'Afternoon', 'Evening', 'Morning, Afternoon', 'Afternoon, Evening', 'Morning, Evening', 'Morning, Afternoon, Evening']
    
    # Generate all possible strict combinations using itertools.product
    all_combinations = list(itertools.product(meeting_frequencies, meeting_days, meeting_times))

    strict_frequencies = ['Weekly (60 minute sessions)', 'Bi-weekly (90 minute sessions)']
    strict_days = ['Weekday', 'Weekend']
    strict_times = ['Morning', 'Afternoon', 'Evening']


    regrouping_map = {'I’m open to either':strict_frequencies,
                      'Weekday, Weekend':strict_days,
                      'Morning, Afternoon': ['Morning', 'Afternoon'],
                      'Afternoon, Evening':[ 'Afternoon', 'Evening'],
                      'Morning, Evening': ['Morning', 'Evening'],
                      'Morning, Afternoon, Evening': ['Morning', 'Afternoon', 'Evening']
                      }

    strict_combinations = list(itertools.product(strict_frequencies, strict_days, strict_times))


    flexibility_combinations = [
        (['I’m open to either'], meeting_days, meeting_times),  # Flexible in meeting frequency
        (meeting_frequencies, ['Weekday, Weekend'], meeting_times),  # Flexible in meeting days
        (meeting_frequencies, meeting_days, ['Morning, Afternoon', 'Afternoon, Evening', 'Morning, Evening'])  # Flexible in time
    ]

    fully_flexible = ('I’m open to either', 'Weekday, Weekend', 'Morning, Afternoon, Evening')
    return strict_combinations, flexibility_combinations, fully_flexible


strict_combinations, flexibility_combinations, fully_flexible = create_combinations()

st.write(fully_flexible)

st.write(strict_combinations)

st.write(flexibility_combinations)