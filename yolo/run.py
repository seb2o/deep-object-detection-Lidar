import time

import matplotlib.pyplot as plt
from ultralytics import YOLO
import pickle
import winsound


def main(project_name: str) -> None:
    model_tuned = YOLO("yolov8s.pt")
    model_tuned.to("cuda")
    model_tuned.train(
        data='../NAPLab-LiDAR/data.yaml',
        epochs=300,
        patience=20,
        batch=4,
        imgsz=1024,
        rect=True,
        save_period=10,
        cache='ram',
        device=0,
        project=project_name,
        deterministic=False,
        plots=True,
        hsv_h=0,
        mosaic=False,
        copy_paste=0.5,
        mixup=0.2,
        flipud=0.25,
        shear=0.1,
        degrees=10
    )
    model_tuned_metrics = model_tuned.val(data="../NAPLab-LiDAR/test.yaml")
    print(model_tuned_metrics)


if __name__ == '__main__':
    print("enter run name : ")
    n = input()
    try:
        main(n)
    finally:
        for _ in range(10):
            winsound.Beep(4000, 500)
            time.sleep(.15)


