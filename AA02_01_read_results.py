import pickle
import os
import numpy as np
import pandas as pd
import support_based as spb

# Function to read pickle files
def read_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# Convert attribute strings to numeric values
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

# Function to get the most recent result files
def get_key_results(path):
    files = [f for f in os.listdir(path) if 'fea_res_' in f and 'beforetrain' not in f and 'notrained' not in f]
    latest_file = max(files, key=lambda x: int(x.split('_')[-1]))

    result_files = ['fea_res_beforetrain', latest_file, 'fea_res_99']
    return [read_pickle(os.path.join(path, f)) for f in result_files]

# Paths
result_path = '../results/'
excel_path = '../data_excel/'
folders = os.listdir(result_path)

# Initialize result storage
all_results = []

# Process each folder
for folder in folders:
    print(f"Processing folder: {folder}")

    # Read the excel file for the current folder
    excel_file_path = os.path.join(excel_path, f"{folder}.xlsx")
    try:
        excel_data = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"Excel file for {folder} not found.")
        continue

    # Filter data for training set
    excel_data = excel_data[excel_data['data_division'] == 'te']

    # Get the result files
    result_list = get_key_results(os.path.join(result_path, folder))

    # Get attributes from the excel data
    attribute_type = folder.split('_')[1].replace('faminc', 'fam_inc')
    attribute_values = [
        excel_data[excel_data['im'] == img_name][attribute_type].values[0]
        for img_name in [x.split('/')[1] for x in result_list[0][0]]
    ]
    attribute_values = convert_attribute_to_numeric(attribute_values, attribute_type)

    # Get the label values
    labels = np.array([x.split('/')[0] for x in result_list[0][0]])
    labels[labels == 'class_A'] = 0
    labels[labels == 'class_B'] = 1
    labels = labels.astype(np.int32)

    # Calculate metrics
    metric_names = ['beforetrain', 'best', 'last']
    metrics = {}

    for idx, result in enumerate(result_list):
        overall_metrics = spb.cal_met_without_opt(result[2], labels)
        group_metrics = []

        # Group-wise metrics
        unique_attributes = np.unique(attribute_values)
        for attribute in unique_attributes:
            group_indices = np.where(attribute_values == attribute)
            group_metrics.append(spb.cal_met_without_opt(result[2][group_indices], labels[group_indices]))

        metrics[metric_names[idx]] = [overall_metrics, np.stack(group_metrics, 0)]

    all_results.append([folder, metrics])

# Save the results to a pickle file
with open('../other_files/res_summary', 'wb') as file:
    pickle.dump(all_results, file)
