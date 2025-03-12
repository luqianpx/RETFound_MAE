import pickle
import os
import numpy as np
import pandas as pd
import support_based as spb

# Function to read pickle file
def read_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Function to convert attribute string to numeric values
def convert_attribute_to_numeric(attribute_values, attribute_type):
    attribute_values = np.array(attribute_values)

    attribute_mapping = {
        'age': {'< 60': 0, '>= 60': 1},
        'sex': {'Female': 0, 'Male': 1},
        'race': {
            'White': 0, 'British': 0, 'Irish': 0, 'Any other white background': 0,
            'Asian or Asian British': 1, 'Indian': 1, 'Pakistani': 1, 'Bangladeshi': 1, 'Any other Asian background': 1,
            'Black or Black British': 2, 'Caribbean': 2, 'African': 2, 'Any other Black background': 2
        },
        'fam_inc': {
            'Less than 18,000': 0, '18,000 to 30,999': 1, '31,000 to 51,999': 2,
            '52,000 to 100,000': 3, 'Greater than 100,000': 4
        }
    }

    if attribute_type in attribute_mapping:
        mapping = attribute_mapping[attribute_type]
        return np.vectorize(lambda x: mapping.get(x, -1))(attribute_values)

    return attribute_values  # If attribute is not found, return original

# Function to read file from the possible directories
def read_excel_from_multiple_dirs(file_name, dirs):
    for dir in dirs:
        file_path = os.path.join(dir, f"{file_name}.xlsx")
        if os.path.isfile(file_path):
            return pd.read_excel(file_path)
    raise FileNotFoundError(f"{file_name}.xlsx not found in any provided directories.")

# Define the paths
result_path = '../results/'
excel_dir_1 = '../data_excel/'
excel_dir_2 = '../data_excel1/'
dirs_to_check = [excel_dir_1, excel_dir_2]

# Initialize the result storage
all_results = []

# Loop through the result directories
for folder in os.listdir(result_path):
    print(f"Processing: {folder}")

    # Read the Excel info file
    try:
        excel_data = read_excel_from_multiple_dirs(folder, dirs_to_check)
    except FileNotFoundError:
        print(f"Error: {folder}.xlsx not found!")
        continue

    excel_data = excel_data[excel_data['data_division'] == 'te']

    # Read the results from pickle file
    result_file_path = os.path.join(result_path, folder, 'fea_res_notrained')
    results = read_pickle(result_file_path)

    # Get the attribute and convert to numeric
    attribute_type = folder.split('_')[1].replace('faminc', 'fam_inc')
    attribute_values = [excel_data[excel_data['im'] == im][attribute_type].values[0] for im in results[0]]
    attribute_values = convert_attribute_to_numeric(attribute_values, attribute_type)

    # Get labels for disease
    disease_column = ['dis1', 'dis2', 'dis3', 'dis4']
    disease_name = ['Diabetes related eye disease', 'Glaucoma', 'Macular degeneration', 'Other serious eye condition']
    label_column = disease_column[folder.split('_')[2]]
    labels = np.array(excel_data[disease_name[disease_column.index(label_column)]]).astype(np.int32)

    # Calculate metrics without optimization
    overall_metrics = spb.cal_met_without_opt(results[2], labels)

    # Group-wise metrics
    group_metrics = []
    unique_attributes = np.unique(attribute_values)
    for attr in unique_attributes:
        group_indices = np.where(attribute_values == attr)
        group_metrics.append(spb.cal_met_without_opt(results[2][group_indices], labels[group_indices]))

    group_metrics = np.stack(group_metrics, 0)

    # Store the results
    all_results.append([folder, overall_metrics, group_metrics])

# Save the results
with open('../other_files/res_summary_untrained', 'wb') as file:
    pickle.dump(all_results, file)
