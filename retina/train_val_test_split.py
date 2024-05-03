import shutil
import os
import numpy as np
import pathlib

file_location = "../NAPLab-LiDAR"
image_path = os.path.join(file_location, "images")
old_label_path = os.path.join(file_location, "old_labels")
new_label_path = os.path.join(file_location, "new_labels")
label_path = new_label_path
split_path = os.path.join(file_location, "splitted")

img_names = list(sorted(os.listdir(image_path)))
txt_names = list(sorted(os.listdir(label_path)))
ge = np.array(list(zip(img_names, txt_names)))

# 19 total scenes
# good val segment: 000302 - 000402
# okayish test segment 001505 - 001604
# 101 images
val = ge[302:403]
test = ge[1505:1605]
g = np.concatenate((ge[:302], ge[403:1505], ge[1605:]))
len(g)


# get 44 extra random pictures
np.random.shuffle(g)
train = g[:-84]
val = np.concatenate((val, g[-44:]))
test = np.concatenate((test, g[-84:-44]))
len(train)

# clear the output
if os.path.exists(split_path) and os.path.isdir(split_path):
    shutil.rmtree(split_path)
pathlib.Path(os.path.join(split_path,"images","train")).mkdir(exist_ok=True, parents=True)
pathlib.Path(os.path.join(split_path,"images","val")).mkdir(exist_ok=True, parents=True)
pathlib.Path(os.path.join(split_path,"images","test")).mkdir(exist_ok=True, parents=True)
pathlib.Path(os.path.join(split_path,"labels","train")).mkdir(exist_ok=True, parents=True)
pathlib.Path(os.path.join(split_path,"labels","val")).mkdir(exist_ok=True, parents=True)
pathlib.Path(os.path.join(split_path,"labels","test")).mkdir(exist_ok=True, parents=True)

for row in train:
    shutil.copy(os.path.join(image_path, row[0]),f"{file_location}/splitted/images/train/{row[0]}")
    shutil.copy(os.path.join(label_path, row[1]),f"{file_location}/splitted/labels/train/{row[1]}")

for row in val:
    shutil.copy(os.path.join(image_path,row[0]),f"{file_location}/splitted/images/val/{row[0]}")
    shutil.copy(os.path.join(label_path,row[1]),f"{file_location}/splitted/labels/val/{row[1]}")

for row in test:
    shutil.copy(os.path.join(image_path,row[0]),f"{file_location}/splitted/images/test/{row[0]}")
    shutil.copy(os.path.join(label_path,row[1]),f"{file_location}/splitted/labels/test/{row[1]}")