import os
img_root = '/Users/charlottekwong/Documents/Deep-Bushfire-Detection/data/training'
boxes_root = '/Users/charlottekwong/Documents/Deep-Bushfire-Detection/data/boxLabels'

lst_img = sorted(f for f in os.listdir(img_root) if f.endswith('.jpg'))
lst_boxes = sorted(f for f in os.listdir(boxes_root) if f.endswith('.txt'))
image_path = list(map(lambda x: os.path.join(img_root, x), lst_img))
boxes_path = list(map(lambda x: os.path.join(boxes_root, x), lst_boxes))

# Read any file and judge if file is empty
for i in range(len(image_path)):
    if os.stat(boxes_path[i]).st_size == 0:
        os.remove(boxes_path[i])
        os.remove(image_path[i])
