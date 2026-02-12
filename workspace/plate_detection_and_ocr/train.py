from ultralytics import YOLO


def main():
    model = YOLO('yolo26x.pt')

    # Check pre-trained model
    print("Before training:")
    print(type(model.names),len(model.names))
    print(model.names)

    # Train the model
    model.train(data="./carplate.yaml",
                epochs=30,
                # patience=3,
                batch=16,
                imgsz=640,
                device=[0, ])

    # Check trained model
    print("\nAfter training:")
    print(type(model.names),len(model.names))
    print(model.names)


if __name__ == "__main__":
    main()
