from ultralytics import YOLO


def main(project_name: str) -> None:
    model_tuned = YOLO("yolov8s.pt")
    model_tuned.to("cuda")
    model_tuned.train(
        data='../NAPLab-LiDAR/data.yaml',
        time=.1,
        patience=3,
        batch=-1,
        imgsz=1024,
        rect=True,
        save_period=10,
        cache='ram',
        device=0,
        project=project_name,
        pretrained=False,
        deterministic=False,
        plots=True,
        hsv_h=0,
        mosaic=False,
        copy_paste=0.5
    )
    model_tuned_metrics = model_tuned.val(data="../NAPLab-LiDAR/test.yaml")
    print(model_tuned_metrics)


if __name__ == '__main__':
    print("enter run name : ")
    n = input()
    main(n)
