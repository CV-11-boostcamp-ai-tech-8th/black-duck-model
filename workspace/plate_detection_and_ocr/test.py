from ultralytics import YOLO

def main():

    model = YOLO('runs/detect/train/weights/best.pt')

    # Check trained model
    print("Inference model:")
    print(type(model.names),len(model.names))
    print(model.names)

    # Inference the model
    metrics = model.val(data="./carplate.yaml",
                    split="test",
                    imgsz=640,
                    visualize=True,
                    save_json=True,
                    device=[0, ])
    
    # Check evaluation metrics
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.map)    # map50-95
    print(metrics.box.maps)   # a list containing mAP50-95 for each category
    
    print(metrics.confusion_matrix.summary())


if __name__ == "__main__":
    main()
