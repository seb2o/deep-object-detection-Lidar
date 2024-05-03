import os
import pathlib
import shutil

import numpy as np
import pandas as pd


def load_boxes(labels_path):
    files = os.listdir(labels_path)
    col_labels = ['classID', 'x', 'y', 'width', 'height']
    dataset_boxes: pd.DataFrame = pd.concat([
        pd.read_csv(
            f"{labels_path}/{filename}",
            sep=' ',
            names=col_labels)
        .reindex(['frame', *col_labels], axis='columns')
        .assign(frame=filename)
        for filename in files
    ], ignore_index=True)

    return dataset_boxes


def remove_class_3(boxes):
    tmp = boxes.copy(deep=True)
    tmp['classID'] = boxes['classID'].apply(lambda c: c if c < 3 else c - 1)
    return tmp


def boxes_to_txt(boxes, dataset_path) -> None:
    g = boxes.groupby('frame')
    for name, group in g:
        group.drop(columns='frame').to_csv(f"{dataset_path}/new_labels/{name}", sep=' ', header=None, index=False)


def sample_names(images_path, labels_path):
    img_names = os.listdir(images_path)
    txt_names = os.listdir(labels_path)
    ge = np.array(list(zip(img_names, txt_names)))
    return ge


def samples_split(name_list):
    val = name_list[302:403]
    test = name_list[1505:1605]

    g = np.concatenate(
        (name_list[:302],
         name_list[403:1505],
         name_list[1605:]))
    np.random.shuffle(g)

    train = g[:-84]

    val = np.concatenate((val, g[-44:]))
    test = np.concatenate((test, g[-84:-44]))

    return train, val, test


def make_splitted_dir(splitted_path) -> None:

    if os.path.exists(splitted_path) and os.path.isdir(splitted_path):
        shutil.rmtree(splitted_path)

    pathlib.Path(os.path.join(splitted_path, "images", "train", )).mkdir(exist_ok=True, parents=True)
    pathlib.Path(os.path.join(splitted_path, "images", "val", )).mkdir(exist_ok=True, parents=True)
    pathlib.Path(os.path.join(splitted_path, "images", "test", )).mkdir(exist_ok=True, parents=True)

    pathlib.Path(os.path.join(splitted_path, "labels", "train", )).mkdir(exist_ok=True, parents=True)
    pathlib.Path(os.path.join(splitted_path, "labels", "val", )).mkdir(exist_ok=True, parents=True)
    pathlib.Path(os.path.join(splitted_path, "labels", "test", )).mkdir(exist_ok=True, parents=True)



def copy_to_splitted(dataset_path):

    splitted_path = os.path.join(dataset_path, "splitted")
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "new_labels")

    train, val, test = samples_split(
        sample_names(
            images_path=images_path,
            labels_path=labels_path
        )
    )

    make_splitted_dir(splitted_path)

    for sample in train:
        shutil.copy(os.path.join(images_path, sample[0]), f"{splitted_path}/images/train/{sample[0]}")
        shutil.copy(os.path.join(labels_path, sample[1]), f"{splitted_path}/labels/train/{sample[1]}")

    for sample in val:
        shutil.copy(os.path.join(images_path, sample[0]), f"{splitted_path}/images/val/{sample[0]}")
        shutil.copy(os.path.join(labels_path, sample[1]), f"{splitted_path}/labels/val/{sample[1]}")

    for sample in test:
        shutil.copy(os.path.join(images_path, sample[0]), f"{splitted_path}/images/test/{sample[0]}")
        shutil.copy(os.path.join(labels_path, sample[1]), f"{splitted_path}/labels/test/{sample[1]}")




def main():

    dataset_path = '/cluster/work/spboo/dlProject/dataset/NAPLab-LiDAR'

    # Create new_labels folder
    boxes_to_txt(
        remove_class_3(
            load_boxes(
                f"{dataset_path}/labels"
            )
        ),
        dataset_path
    )

    copy_to_splitted(dataset_path)



if __name__ == '__main__':
    main()

