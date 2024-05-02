import time

from ultralytics import YOLO
import winsound


def main(project_name: str) -> None:
    model_tuned = YOLO("yolov8s.pt")
    model_tuned.to("cuda")
    model_tuned.train(
        data='../NAPLab-LiDAR/data.yaml',
        epochs=20,
        patience=20,
        batch=4,
        imgsz=640,
        rect=False,
        save_period=10,
        cache='ram',
        device=0,
        project=project_name,
        deterministic=False,
        plots=True,
        hsv_h=0,
        mosaic=False,
        copy_paste=0,
        mixup=0,
        flipud=0,
        shear=0,
        degrees=0
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


