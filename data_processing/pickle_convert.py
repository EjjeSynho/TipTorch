#%%
import os
import pickle5 as pickle5
import pickle
import numpy as np

# filepath = 'C:/Users/akuznets/Data/SPHERE/test/12_SPHER.2016-09-20T06.43.02.203IRD_FLUX_CALIB_CORO_RAW_left.pickle'
# output_filepath = 'C:/Users/akuznets/Data/SPHERE/test_fits/12_SPHER.2016-09-20T06.43.02.203IRD_FLUX_CALIB_CORO_RAW_left.json'

def convert_pickles_to_json(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is a pickle file
        if filename.endswith('.pickle'):
            filepath = os.path.join(input_folder, filename)

        # Load the pickle file containing the dictionary
        with open(filepath, 'rb') as f: data = pickle5.load(f)

        # write pickle file to pickle
        output_filepath = os.path.join(output_folder, filename)
        with open(output_filepath, 'wb') as file: pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

convert_pickles_to_json('C:/Users/akuznets/Data/SPHERE/test/', 'C:/Users/akuznets/Data/SPHERE/test_new/')

# %%
import matplotlib.pyplot as plt

input_folder = 'C:/Users/akuznets/Data/SPHERE/test_new/'

filenames =  os.listdir(input_folder)

filepath = os.path.join(input_folder, filenames[0])



with open(filepath, 'rb') as f: data = pickle.load(f)


