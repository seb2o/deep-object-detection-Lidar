import torchvision
import torch
import os
import time
import numpy as np
from PIL import Image

file_location = "../NAPLab-LiDAR"
image_path = os.path.join(file_location, "images")
old_label_path = os.path.join(file_location, "old_labels")
new_label_path = os.path.join(file_location, "new_labels")
label_path = new_label_path
split_path = os.path.join(file_location, "splitted")

print(torchvision.__version__)
retinanet = torchvision.models.detection.retinanet_resnet50_fpn(
    weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_epochs = 1
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
            boxes.append([xmin, ymin, xmax, ymax])
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
validation_dataset = NAPLabLoader("val")

data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)
test_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=2, collate_fn=collate_fn)

# parameters
params = [p for p in retinanet.parameters() if p.requires_grad]  # select parameters that require gradient calculation
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

len_dataloader = len(data_loader)

# about 4 min per epoch on Colab GPU
for epoch in range(num_epochs):
    start = time.time()
    retinanet.train()

    i = 0
    epoch_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = retinanet(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        i += 1

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses
    print(epoch_loss, f'time: {time.time() - start}')
