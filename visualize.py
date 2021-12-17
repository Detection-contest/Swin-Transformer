import pandas as pd
from mmdet.apis import inference_detector, init_detector, show_result_pyplot, async_inference_detector
import asyncio
import os

df = pd.read_csv('dataset/ImageSets/Main/val.txt', names=['jpg'])

out_dir = "output"
if (os.path.isdir(out_dir) == False):
    make_out_dir = out_dir
    os.makedirs(make_out_dir)

visual_root = (out_dir + "/swin_VisualImage")
if (os.path.isdir(visual_root) == False):
    make_visual_dir = visual_root
    os.makedirs(make_visual_dir)

image_root = "dataset/JPEGImages/"
point_root = "Swin-Transformer-Object-Detection/work_dirs/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/"

# only epoch 14
config_file = (point_root + 'cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py')
checkpoint_file = (point_root + 'epoch_14.pth')

model = init_detector(config_file, checkpoint_file, device='cuda:0')

def async_main():
    for j in range(0, 100):
        img = (image_root + df.loc[j]['jpg'] + '.jpg')
        result = inference_detector(model, img)
        model.show_result(img, result, out_file=(visual_root + '/epoch_14_' + df.loc[j]['jpg'] + '.jpg'))


if __name__ == '__main__':
    async_main()
