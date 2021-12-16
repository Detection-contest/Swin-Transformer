import os
import numpy as np
import shutil
from tqdm import tqdm

os.makedirs("coco_data/Test")

root_dir = 'test_data/'

for all_folder in tqdm(os.listdir(root_dir)):
    sub_dir = root_dir + '/' + all_folder
    for sub_folder in os.listdir(sub_dir):
        file_dir = root_dir + '/' + all_folder + '/' + sub_folder
        for file in os.listdir(file_dir):
            file_name = file_dir + '/' + file
            shutil.copy(file_name, "coco_data/Test/")
            
            
f_test = open("coco_data/test.txt", 'w')

root_dir = 'coco_data/Test/'
TestFileNames = os.listdir(root_dir)

for name in TestFileNames:
    f_test.write(name.rstrip('.jpg') + '\n')
f_test.close()

print('finished')