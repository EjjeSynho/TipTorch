#%%
%reload_ext autoreload
%autoreload 2

import os
from project_settings import SPHERE_DATA_FOLDER


with open(os.path.join(SPHERE_DATA_FOLDER, 'labels_old.txt'), 'r') as file:
    lines = file.readlines()

new_files = os.listdir(os.path.join(SPHERE_DATA_FOLDER, 'IRDIS_images'))

#%%
labels_split = lambda x: set(x.split(', '))

def strip_filename(x, ext='.png'):
    x_ = x.strip()
    underscore_loc = x_.strip().find('_')
    file_ext_loc   = x_.strip().find(ext)

    id = x_[:underscore_loc]
    spitted_line = x_[underscore_loc+1:file_ext_loc]
    return int(id), spitted_line

def extract_filename(line):
    line_ = line.strip()
    semicolon_loc = line_.find(':')
    return line_[:semicolon_loc], labels_split(line_[semicolon_loc+2:].strip())

#%%
new_files_id_data, old_file_id_data = {}, {}

for file in new_files:
    id, pure_name = strip_filename(file)
    new_files_id_data[pure_name] = id
    
# Sort dictionary based on values
new_files_id_data = dict(sorted(new_files_id_data.items(), key=lambda item: item[1]))

for line in lines:
    file, labels  = extract_filename(line)
    id, pure_name = strip_filename(file)

    if pure_name in old_file_id_data:
        print(f'Duplicate found: {pure_name}')
        labels = old_file_id_data[pure_name][1] | labels      

    old_file_id_data[pure_name] = tuple([id, labels])

#%%
new_labels = {}

for key, value in new_files_id_data.items():
    if key in old_file_id_data:
        labels_str = ', '.join(list(old_file_id_data[key][1]))
        # new_labels[value] = f'{old_file_id_data[key][0]}_{key}.png: {labels_str}'
        new_labels[value] = f'{value}_{key}.png: {labels_str}'
    else:
        new_labels[value] = f'{value}_{key}.png: '


#%%
with open(os.path.join(SPHERE_DATA_FOLDER, 'labels_new.txt'), 'w') as file:
    for _, line in new_labels.items():
        file.write(line+'\n')


#%%
from tqdm import tqdm
import os
from collections import defaultdict
from datetime import datetime

def get_duplicates_info(raw_dir):
    # Dictionary to track file occurrences using file names
    file_map = defaultdict(list)

    # Scan through all subdirectories in the raw_dir
    for root, _, files in tqdm(os.walk(raw_dir)):
        for file in files:
            file_map[file].append(root)

    # Identify duplicates (files that appear in more than one directory)
    duplicates = {file: paths for file, paths in file_map.items() if len(paths) > 1}

    # Build the duplicates dictionary with file info
    duplicates_dict = {}
    for file, directories in duplicates.items():
        duplicates_dict[file] = []
        for directory in directories:
            file_path = os.path.join(directory, file)
            # Get the last modification date
            mod_time = os.path.getmtime(file_path)
            mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            duplicates_dict[file].append({
                "path": file_path,
                "last_modified": mod_date
            })
    return duplicates_dict


#%
# Define the directory to scan
# raw_dir = os.path.join(SPHERE_DATA_FOLDER, 'IRDIS_RAW', 'SPHERE_DC_DATA_LEFT')
raw_dir = os.path.join(SPHERE_DATA_FOLDER, 'IRDIS_RAW', 'SPHERE_DC_DATA_RIGHT')

# Get the duplicates information as a dictionary
duplicates_info = get_duplicates_info(raw_dir)


#%%
def get_earliest_file_path(entries):
    """
    Given a list of dictionaries with keys 'path' and 'last_modified' (formatted as YYYY-MM-DD HH:MM:SS),
    return the file path corresponding to the earliest modification date.
    """
    # Use min with a key that parses the modification date string into a datetime object
    earliest_entry = min(entries, key=lambda x: datetime.strptime(x["last_modified"], '%Y-%m-%d %H:%M:%S'))
    return earliest_entry["path"]

# Example usage
files_to_remove = []
for entries in duplicates_info.values():
    files_to_remove.append( get_earliest_file_path(entries) )
 
#%%
# Remove files
for file_path in tqdm(files_to_remove):
    os.remove(file_path)
    

#%%
# Delete empty folders
def delete_empty_folders(directory):
    for root, dirs, files in tqdm(os.walk(directory, topdown=False)):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):  # Check if the directory is empty
                os.rmdir(dir_path)  # Remove the empty directory
                print(f"Removed empty directory: {dir_path}")

# Call the function to delete empty folders in the specified directory
delete_empty_folders(raw_dir)


