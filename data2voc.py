import os
import numpy as np
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET

os.makedirs("coco_data")
os.makedirs("coco_data/Annotations")
os.makedirs("coco_data/Images")

root_dir = 'data'

for all_folder in tqdm(os.listdir(root_dir)):
    sub_dir = root_dir + '/' + all_folder
    for sub_folder in os.listdir(sub_dir):
        file_dir = root_dir + '/' + all_folder + '/' + sub_folder
        for file in os.listdir(file_dir):
            file_name = file_dir + '/' + file
            if 'xml' in file:
                shutil.copy(file_name, "coco_data/Annotations/")
            else:
                shutil.copy(file_name, "coco_data/Images/")
                
root_dir = 'coco_data/Annotations/'

for filename in os.listdir(root_dir):
    file = root_dir + '/' + filename
    os.rename(file, file.replace('_v001_1', ''))
    
root_dir = 'coco_data/Annotations/'
image_dir = 'coco_data/Images/'

for xml_name in tqdm(os.listdir(root_dir)):
    tree = ET.parse(root_dir + xml_name)
    root = tree.getroot()
    
    # 없는 object 객체 지우기
    if len(root.findall('object')) == 0:
        os.remove(root_dir + xml_name)
        os.remove(image_dir + xml_name.replace(".xml",".jpg"))
        continue
    
    # 없는 line 객체 지우기
    if root.findall('line'):
        os.remove(root_dir + xml_name)
        print(root_dir + xml_name)
        os.remove(image_dir + xml_name.replace(".xml",".jpg"))
        print(image_dir + xml_name.replace(".xml",".jpg"))
        continue
    
    for obj in root.findall('object'):
        if obj.find('name').text is None:
            root.remove(obj)
            continue
        
        # Bounding Box좌표 계산 
        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text)
        y1 = int(xmlbox.find('ymin').text)
        x2 = int(xmlbox.find('xmax').text)
        y2 = int(xmlbox.find('ymax').text)
        
        # Bounding Box좌표의 min, max값 다를 경우 바꾸기
        if x2 < x1 or y2 < y1:
            root.remove(obj)
            continue
        
        name = obj.find('name').text
        
        if name.startswith('Vehicle'):
            obj.find('name').text = 'Vehicle'
        elif name.startswith('Pedestrian'):
            obj.find('name').text = 'Pedestrian'
        elif name.startswith('TrafficLight'):
            obj.find('name').text = 'TrafficLight'
        elif name.startswith('TrafficSign'):
            obj.find('name').text = 'TrafficSign'
        else:
            root.remove(obj)
    tree.write(root_dir + xml_name)
    
f_train = open("coco_data/train.txt", 'w')
f_val = open("coco_data/val.txt", 'w')

root_dir = 'coco_data/Images/'
allFileNames = os.listdir(root_dir)
_, val_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.9)])

train_FileNames = allFileNames
val_FileNames = [name for name in val_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))

for name in train_FileNames:
    f_train.write(name.rstrip('.jpg') + '\n')
f_train.close()

for name in val_FileNames:
    f_val.write(name.rstrip('.jpg') + '\n')
    print(name)
f_val.close()

print('finished')