import time
from ultralytics import YOLO
import winsound


def main(project_name: str) -> None:

    model_tuned = YOLO("yolov8s.yaml")
    model_tuned.to("cuda")

    model_tuned.train(
        data='../NAPLab-LiDAR/data.yaml',
        epochs=300,
        patience=25,
        batch=-1,
        imgsz=1024,
        pretrained=False,
        rect=False,
        save_period=20,
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


if __name__ == '__main__':
    print("enter run name : ")
    n = input()
    try:
        main(n)
    finally:
        for _ in range(20):
            winsound.Beep(4000, 500)
            time.sleep(.15)


