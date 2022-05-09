import torch
#from data import instancedata, binary_mask_to_polygon, NumpyEncoder
from data import instancedata, NumpyEncoder
import datetime
import numpy as np
#from pycocotools import mask
import json

# box format:[xmin, ymin, xmax, ymax]  transfer to [xmin, ymin, w, h] in code

img_root = '/Users/charlottekwong/Documents/Deep-Bushfire-Detection/data/training'
# boxes_root = '/data/local_userdata/tianxinyu/COME_instance/boxes/'
# masks_root = '/data/local_userdata/tianxinyu/COME_instance/masks/'

boxes_root = '/Users/charlottekwong/Documents/Deep-Bushfire-Detection/data/boxLabels'
#masks_root = '/data/local_userdata/tianxinyu/COME_instance_ranking/multi_instance_v2/mask_labels/label5/'
json_result_file = 'cocoFormat'

coco_label = {}
info_content = []
info_content.extend(
    [
        {
            "contributor": "Xinyu Tian",
            "date_created": "2022_02_18",
            "description": "COME_train_dataset_multi_instances_label5_COCO_format",
            "url": "https://github.com/JingZhang617/cascaded_rgbd_sod",
            "version": "1.0",
            "year": "2022"
        }
    ]
)
coco_label["info"] = info_content

licen_content = []
licen_content.extend(
    [
        {
            "id": 1,
            "name": "license",
            "url": "https://github.com/JingZhang617/cascaded_rgbd_sod"
        }
    ]
)
coco_label["licenses"] = licen_content

cate_content = []
cate_content.extend(
    [
        {
            "id": 1,
            "name": "instance_salient_object",
            "supercategory": "instance_salient_object"
        }
    ]
)
coco_label["categories"] = cate_content

images_list = []
annotations_list = []

#d_train = instancedata(img_root, boxes_root, masks_root)
d_train = instancedata(img_root, boxes_root)
instance_num = 1
for i, (img, target) in enumerate(d_train):
    image_name = target['name']
    image_id = target['image_id']
    bbox = target['boxes']
    #segm = target['masks']

    C, H, W = img.shape
    # print(img.shape)

    #assert segm.max() == 1

    images_list.extend(
        [
            {
                "id": image_id.item(),
                "file_name": image_name,
                "width": W,
                "height": H,
                "date_capture": datetime.datetime.utcnow().isoformat(' '),
                "licenses": 1,
                "coco_url": "",
                "flicker_url": ""
            }
        ]
    )
    print(image_name)

    # print(len(bbox))
    for k in range(len(bbox)):
        #segmap = segm[k].astype(np.uint8)
        #seg_polygon = binary_mask_to_polygon(segmap, tolerance=1)
        #assert segmap.shape == (H, W)
        #binary_mask_encoded = mask.encode(np.asfortranarray(segmap))
        # print(binary_mask_encoded)
        #area = mask.area(binary_mask_encoded)
        # print(len(seg_polygon[0]))
        # print(area)
        bbox_xyxy = bbox[k]
        bbox_xywh = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2]-bbox_xyxy[0], bbox_xyxy[3]-bbox_xyxy[1]]
        area = bbox_xywh[2] * bbox_xywh[3]
        annotations_list.extend(
            [
                {
                    "id": instance_num,
                    "image_id": image_id.item(),
                    #"category_id": class_label[k],
                    "category_id": 1,
                    "iscrowd": 0,
                    "area": area,
                    "bbox": bbox_xywh,
                    #"segmentation": [],
                    "ignore": 0,
                    "width": W,
                    "height": H
                }
            ]
        )
        instance_num += 1

print(i)
print(instance_num)
coco_label["images"] = images_list
coco_label["annotations"] = annotations_list

with open(json_result_file+'label5.json', "w+") as f:
    json.dump(coco_label, f, cls=NumpyEncoder)         # encode json data









