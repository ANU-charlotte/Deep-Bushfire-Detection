import os
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
#from skimage import measure
import json


class instancedata(data.Dataset):
    def __init__(self, img_root, boxes_root):
        lst_img = sorted(f for f in os.listdir(img_root) if f.endswith('.jpg'))
        lst_boxes = sorted(f for f in os.listdir(boxes_root) if f.endswith('.txt'))

        self.img_abbr, self.box_abbr = lst_img[0].split('.')[-1], lst_boxes[0].split('.')[-1] # suffix
        img_list, boxes_list = [], []
        for name in lst_img:
            img_name = name.split('.')[0]
            if img_name + '.' + self.img_abbr in lst_img:
                img_list.append(name)

        for name in lst_boxes:
            box_name = name.split('.')[0]
            if box_name + '.' + self.box_abbr in lst_boxes:
                boxes_list.append(name)

        # for name in lst_masks:
        #     mask_name = name.split('.')[0]
        #     if mask_name + '.' + self.mask_abbr in lst_masks:
        #         masks_list.append(name)

        self.img_transform = transforms.Compose([
            transforms.ToTensor()])

        self.image_path = list(map(lambda x: os.path.join(img_root, x), img_list))
        self.boxes_path = list(map(lambda x: os.path.join(boxes_root, x), boxes_list))
        print(self.boxes_path)
        #self.masks_path = list(map(lambda x: os.path.join(masks_root, x), masks_list))

    def __getitem__(self, item):
        img = Image.open(self.image_path[item]).convert('RGB')
        img = self.img_transform(img)
        with open(self.boxes_path[item], 'r') as f:

            i = 0
            for line in f.readlines():
                line = line.strip('\n').split(' ')
                box = torch.Tensor(list(map(float, line))).unsqueeze(0)
                if i == 0:
                    boxes = box
                    i = 1
                else:
                    boxes = torch.cat((boxes, box), dim=0)

        boxes = boxes.numpy().tolist()
        #masks = np.load(self.masks_path[item])
        #print(masks.max())

        # masks = masks / 255

        #labels = np.ones(masks.shape[0], dtype=int)
        #labels = np.ones(boxes.shape[0], dtype=int)

        name = self.image_path[item].split('/')[-1]
        image_id = name.split('.')[0].split('_')[-1]
        image_id = torch.tensor([int(image_id)])

        #assert len(boxes)==masks.shape[0]
        #assert len(boxes)==labels.shape[0] # check masks and box have same shape

        target = {
            'name': name,
            'image_id': image_id,
            #'labels': labels,
            'boxes': boxes
            #'masks': masks
        }

        return img, target

    def __len__(self):
        return len(self.image_path)

# def close_contour(contour):
#     if not np.array_equal(contour[0], contour[-1]):
#         contour = np.vstack((contour, contour[0]))
#     return contour

#
# def binary_mask_to_polygon(binary_mask, tolerance=0):
#     """Converts a binary mask to COCO polygon representation
#     Args:
#         binary_mask: a 2D binary numpy array where '1's represent the object
#         tolerance: Maximum distance from original points of polygon to approximated
#             polygonal chain. If tolerance is 0, the original coordinate array is returned.
#     """
#     polygons = []
#     # pad mask to close contours of shapes which start and end at an edge
#     padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
#     contours = measure.find_contours(padded_binary_mask, 0.5)
#     contours = np.subtract(contours, 1)
#     for contour in contours:
#         contour = close_contour(contour)
#         contour = measure.approximate_polygon(contour, tolerance)
#         if len(contour) < 3:
#             continue
#         contour = np.flip(contour, axis=1)
#         segmentation = contour.ravel().tolist()
#         # after padding and subtracting 1 we may get -0.5 points in our segmentation
#         segmentation = [0 if i < 0 else i for i in segmentation]
#         polygons.append(segmentation)
#
#     return polygons


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

