import os
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
from os.path import join
import json
from PIL import Image
from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
from tqdm import tqdm

# coco classes
classes = {0 : 'Vehicle', 1 : 'TrafficLight', 2 : 'TrafficSign', 3 : 'Pedestrian'}

# converts the normalized positions  into integer positions
def unconvert(width, height, x, y, w, h):
    
    xmax = int(x+w)
    xmin = int(x)
    ymax = int(y+h)
    ymin = int(y)
    
    return [xmin, ymin, xmax, ymax]

def to_xml(input_json):
    
    with open(input_json, "r") as st_json:
        json_anno = json.load(st_json)

        input_image_dir = "coco_data/Images/" # "inference input image dir/"

        out_dir = "output"
        if (os.path.isdir(out_dir) == False):
            make_out_dir = out_dir
            os.makedirs(make_out_dir)
        
        root_dir = (out_dir + "/inference") # "inference xml 저장 dir/"
        if (os.path.isdir(root_dir) == False):
            make_root_dir = root_dir
            os.makedirs(make_root_dir)

        # 이미지 폴더에서 이미지 output/20200602_140054/1 에 저장하는 것도 해야함
        for i in tqdm(range (len(json_anno))): # images dic 하나씩
            anno = json_anno[i]
            image_name = anno['file_name']
            width = anno['width']
            height = anno['height']

            # ex) output/20200602_140054 없으면
            image_dir = root_dir + "/" + image_name[2:-11]
            if (os.path.isdir(image_dir) == False):
                make_image_dir = image_dir
                os.makedirs(make_image_dir)

            # ex) output/20200602_140054/1
            images_dir = image_dir + "/" + image_name[:1]
            if (os.path.isdir(images_dir) == False):
                make_images_dir = images_dir
                os.makedirs(make_images_dir)

            # 해당 image save
            image_open = Image.open(input_image_dir + image_name)
            image_open.save(images_dir + "/" + image_name)

            # ex) output/20200602_140054/1_Result
            anno_dir = image_dir + "/" + image_name[:2] + "Result"
            if (os.path.isdir(anno_dir) == False):
                make_anno_dir = anno_dir
                os.makedirs(make_anno_dir)

            xml_file_name = anno_dir + "/" + image_name[:-4] + "_v001_1.xml"

            if (os.path.isfile(xml_file_name) == False): # xml이 없는 경우
                node_root = Element('annotation')

                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = (classes[anno['category_id']])

                bbox = anno['bbox']
                new_label = unconvert(width, height, bbox[0], bbox[1], bbox[2], bbox[3])

                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(new_label[0])
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(new_label[1])
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text =  str(new_label[2])
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(new_label[3])

                tree = ElementTree(node_root)
                tree.write(xml_file_name)
                    
            else: # 있는 경우
                targetXML = open(xml_file_name, 'rt', encoding='UTF8')
                tree = ET.parse(targetXML)
                root = tree.getroot()
                
                bbox = anno['bbox']
                new_label = unconvert(width, height, bbox[0], bbox[1], bbox[2], bbox[3])

                node_object = SubElement(root,'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = (classes[anno['category_id']])

                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(new_label[0])
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(new_label[1])
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text =  str(new_label[2])
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(new_label[3])
                    
                root.append(node_object)
                tree.write(xml_file_name)
            
            
if __name__ == '__main__':
    input_json = "result_json.bbox.json"
    to_xml(input_json)