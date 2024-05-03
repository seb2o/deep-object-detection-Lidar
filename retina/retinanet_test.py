import torchvision
import torch
import os
import time
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

###
#   USER DEFINED VARIABLES
###
num_classes = 8
batch_size = 2
num_epochs = 160
image_x = 1024
image_y = 128
lr = 0.0020
momentum = 0.4
weight_decay = 0.0007
prediction_confidence = 0.25
print(f"classes: {num_classes} batch_size: {batch_size} epochs: {
      num_epochs} lr: {lr} momentum: {momentum} weight_decay: {weight_decay}")
####################################################################################

file_location = "../../NAPLab-LiDAR"
image_path = os.path.join(file_location, "images")
old_label_path = os.path.join(file_location, "old_labels")
new_label_path = os.path.join(file_location, "new_labels")
label_path = new_label_path
split_path = os.path.join(file_location, "splitted")

print(torchvision.__version__)
retinanet = torchvision.models.detection.retinanet_resnet50_fpn_v2(
    weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)


# replace classification layer
out_channels = retinanet.head.classification_head.conv[0].out_channels
num_anchors = retinanet.head.classification_head.num_anchors
retinanet.head.classification_head.num_classes = num_classes

cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code
# assign cls head to model
retinanet.head.classification_head.cls_logits = cls_logits

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


retinanet.to(device)


def generate_target(file):
    with open(file) as file:
        boxes = []
        labels = []
        for line in file:
            # coords in %
            if len(line.strip().split(" ")) != 5:
                continue
            class_nbr, centerX, centerY, width, height = line.strip().split(" ")
            xmin = float(centerX) - float(width) / 2.0
            xmax = xmin + float(width)
            ymin = float(centerY) - float(height) / 2.0
            ymax = ymin + float(height)
            boxes.append([round(xmin * image_x), round(ymin * image_y), round(xmax * image_x), round(ymax * image_y)])
            labels.append(int(class_nbr))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return target


class NAPLabLoader(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.img_path = os.path.join(split_path, "images", path)
        self.label_path = os.path.join(split_path, "labels", path)
        self.imgs = list(sorted(os.listdir(self.img_path)))
        self.labels = list(sorted(os.listdir(self.label_path)))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        file_image = self.imgs[idx]
        file_label = self.labels[idx]
        if not file_image[:-3] == file_label[:-3]:
            print(file_image)
            print(file_label)
            assert file_image[:-3] == file_label[:-3]

        img_path = os.path.join(self.img_path, file_image)
        label_path = os.path.join(self.label_path, file_label)

        img = Image.open(img_path).convert("RGB")
        target = generate_target(label_path)

        to_tensor = torchvision.transforms.ToTensor()

        if self.transform:
            img, transform_target = self.transform(np.array(img), np.array(target['boxes']))
            target['boxes'] = torch.as_tensor(transform_target)

        # change to tensor
        img = to_tensor(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


train_dataset = NAPLabLoader("train")
validation_dataset = NAPLabLoader("test")

data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=2, collate_fn=collate_fn)

retinanet.load_state_dict(torch.load(f'retinanet_{num_epochs}.pt', map_location=torch.device('cpu')))

device = torch.device('cpu')
retinanet.to(device)


def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)):
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']):
            if score > threshold:  # select idx which meets the threshold
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds


map50_list = []
for pdx in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    labels = []
    preds_adj_all = []
    annot_all = []

    for im, annot in tqdm(test_data_loader, position=0, leave=True):
        im = list(img.to(device) for img in im)
        annot = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in annot]

        for t in annot:
            labels += t['labels']

        with torch.no_grad():
            preds_adj = make_prediction(retinanet, im, prediction_confidence)
            preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)
            annot_all.append(annot)

    import utils_ObjectDetection as utils
    sample_metrics = []
    for batch_i in range(len(preds_adj_all)):
        sample_metrics += utils.get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=pdx)

    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]  # 배치가 전부 합쳐짐
    precision, recall, AP, f1, ap_class = utils.ap_per_class(
        true_positives, pred_scores, pred_labels, torch.tensor(labels))
    mAP = AP.mean()
    print(f'mAP{pdx} : {mAP}')
    print(f' AP{pdx} : {AP}')
    map50_list.append(mAP)

print(f"map50-95: {np.mean(np.array(map50_list))}")
