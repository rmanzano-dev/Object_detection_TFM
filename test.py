import os
label_path = "NuScenes_YOLO\\train\\labels"
counter = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    "counter": 0
}
for label in os.listdir(label_path):
    counter["counter"] += 1
    with open(os.path.join(label_path, label), 'r', encoding="utf-8") as f:
        for line in f:
            # Parse YOLOv8 format: <class_id> <x_center> <y_center> <width> <height>
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            counter[class_id] += 1

print(counter)