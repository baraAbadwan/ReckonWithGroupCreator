import pandas as pd
import string
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
