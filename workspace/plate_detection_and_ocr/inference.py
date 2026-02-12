import os

from ultralytics import YOLO
from datetime import datetime

def main():

    model = YOLO('runs/detect/train/weights/best.pt')

    # Check trained model
    print("Inference model:")
    print(type(model.names),len(model.names))
    print(model.names)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"predict_plate_{timestamp}"

    # Inference the model
    image_dir = "./car_data/test/images"
    assert os.path.isdir(image_dir), f"Invalid directory: {image_dir}"

    results = model(source=image_dir,
                    # conf=0.25,
                    imgsz=640,  # (1080, 1920),  # 640, 960, 1280
                    save=True,
                    save_txt=True,
                    save_conf=False,
                    save_crop=True,
                    project=project_name,
                    name="predict_plate",   # runs/detect/predict_plate/
                    device=[0, ])

    # Print results
    print(f"\nInference completed for directory: {image_dir}")
    print(f"Saved to runs/detect/{project_name}/")


if __name__ == "__main__":
    main()
