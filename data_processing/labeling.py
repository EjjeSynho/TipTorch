#%%
import os
import pickle
import re
from copy import deepcopy

from project_globals import SPHERE_DATA_FOLDER

def purify_filename(filename):
    pure_filename = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r'^(\d+)_', pure_filename)
    if match:
        number = match.group(1)
        pure_filename = re.sub(r'^\d+_', '', pure_filename)
    else:
        number = None
    return number, pure_filename

invalid_folder = 'C:/Users/akuznets/Data/SPHERE/test_invalid/'
invalid_list = [purify_filename(file) for file in os.listdir(invalid_folder)]
ids_invalid = [i[0] for i in invalid_list]
filenames_invalid = [i[1] for i in invalid_list]

valid_folder = 'C:/Users/akuznets/Data/SPHERE/test/'
valid_list = [purify_filename(file) for file in os.listdir(valid_folder)]
ids_valid = [i[0] for i in valid_list]
filenames_valid = [i[1] for i in valid_list]

#%%
labels_list = 'C:/Users/akuznets/Data/SPHERE/labels.txt'
label_data = []

with open(labels_list, 'r') as f:
    labels = f.readlines()
    for label in labels:
        label_data.append(label.split())        
data = deepcopy(label_data)

#%%
for i,line in enumerate(label_data):
    if line[0] in ids_valid:
        index = ids_valid.index(line[0])
        data[i][0] = filenames_valid[index]
    elif line[0] in ids_invalid:
        index = ids_invalid.index(line[0])
        data[i][0] = filenames_invalid[index]
    else:
        print('File not found: ', line[0])

# %%
label_list = []
for line in data:
    for i in line[1:]:
        label_list.append(i)
label_list = set(label_list)

#%%
newly_reduced = os.listdir(SPHERE_DATA_FOLDER + 'IRDIS_reduced/')

newly_reduced_files = [purify_filename(file)[1] for file in newly_reduced]
newly_reduced_ids   = [purify_filename(file)[0] for file in newly_reduced]

# Add ids in front of the filenames
for i,line in enumerate(data):
    filename = line[0]
    if filename in newly_reduced_files:
        index = newly_reduced_files.index(filename)
        data[i][0] = newly_reduced_ids[index] + '_' + filename

# %%
label_dict = {}
for i,label in enumerate(label_list):
    for line in data:
        if label in line:
            if label not in label_dict:
                label_dict[label] = []
            label_dict[label].append(line[0])

no_corono = []
for line in data:
    line[0] = line[0]
    if 'CALIB_NO_CORO_RAW' in line[0]:
        no_corono.append(line[0])

# Add new label to the dictionary
label_dict['no coronograph'] = no_corono


#%%
import json

with open(SPHERE_DATA_FOLDER+'labels.json', 'w') as f:
    json.dump(label_dict, f)

#%%
# Read json file
with open(SPHERE_DATA_FOLDER+'labels.json', 'r') as f:
    label_dict_0 = json.load(f)

#%%

images_valid_dir   = SPHERE_DATA_FOLDER+'reduced_imgs/'
images_invalid_dir = SPHERE_DATA_FOLDER+'invalid_imgs/'

images_valid = os.listdir(images_valid_dir)

# move bad files to another folder
import shutil

for file in label_dict['Corrupted']:
    shutil.move(images_valid_dir + file + '.png', images_invalid_dir + file + '.png')
    
#%%

labels_lists = ['doubles.txt', 'invalid.txt', 'Class A.txt', 'Class B.txt', 'Class C.txt', 'LWE.txt']

label_dict = {}
for label_list in labels_lists:
    # Read file
    with open(SPHERE_DATA_FOLDER+'images/' + label_list, 'r') as f:
        files = f.readlines()
        files = [filename.rstrip() for filename in files]

    # Add new label to the dictionary
    label_dict[label_list[:-4]] = files

#%%


label_dict['Blurry'] = label_dict_0['Blurry']
label_dict['Noisy'] = label_dict_0['Noisy']
label_dict['Clipped'] = label_dict_0['Clipped']
label_dict['Crossed'] = label_dict_0['Crossed']
label_dict['Streched'] = label_dict_0['Streched']
label_dict['Wingsgone'] = label_dict_0['Wingsgone']
label_dict['No coronograph'] = label_dict_0['No coronograph']

#%%

# Remove file extension
for key in label_dict:
    label_dict[key] = [filename.replace('.png','') for filename in label_dict[key]]

