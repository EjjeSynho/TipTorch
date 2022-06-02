#%%
from os import path
import sys
import os

path_base = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\'

script = open('TipToy.py')
code = script.read()
files = os.listdir(path_base)

for file in files:
    path_data = path.join(path_base,file)
    sys.argv = [path_data, path_data]

    try:
        exec(code) 
    except RuntimeError:
        print("Oops! Anyway...")



# %%
