import os

# Define the base path for the results
pa = '../results/'
fo_li = os.listdir(pa)

# List of file names to check
file_names = ['fea_res_99', 'fea_res_0', 'fea_res_beforetrain']

# Function to check for missing files
def check_missing_files(file_name):
    missing_files = []
    print(f'No {file_name} --------------------------------------------------------------------------------------------------------------------')
    num = 0
    for fo in fo_li:
        file_path = os.path.join(pa, fo, file_name)
        if not os.path.isfile(file_path):
            num += 1
            missing_files.append(fo)
            print(num, fo)
    return missing_files

# Check for missing files for each type
for file_name in file_names:
    check_missing_files(file_name)
