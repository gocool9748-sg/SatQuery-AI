from ultralytics import YOLO
from pathlib import Path


# DOTA satellite / aerial object detection model
model = YOLO("yolo26n-obb.pt")


def detect_satellite_objects(image_path):

    results = model.predict(
        source=image_path,
        conf=0.25,
        imgsz=1024,
        save=True
    )

    return results


if __name__ == "__main__":

    image_path = input("Enter image path: ").strip()

    if not Path(image_path).exists():
        print("❌ Image not found.")
        exit()

    print("\n🛰️ Running DOTA satellite object detection...\n")

    results = detect_satellite_objects(image_path)

    total_objects = 0

    for result in results:

        if result.obb is None:
            print("No objects detected.")
            continue

        for i in range(len(result.obb.cls)):

            class_id = int(result.obb.cls[i])
            confidence = float(result.obb.conf[i])

            class_name = result.names[class_id]

            total_objects += 1

            print(
                f"🔹 {class_name} "
                f"| confidence: {confidence:.2f}"
            )

    print("\n--------------------------------")
    print(f"🛰️ Total objects detected: {total_objects}")
    print("--------------------------------")

    print("\n✅ Detection complete.")
    print("📁 Check the runs/obb/ folder for the annotated image.")