from ultralytics import YOLO
import os
from argparser import get_args
import json
import time
args = get_args()
# Load YOLO model, depending if pretrained or not
if args.pretrained:
    model = YOLO('yolov8n.pt')  # Puedes usar otros modelos como 'yolov8s.pt', 'yolov8m.pt', etc.
else:
    model = YOLO('yolov8n.yaml')  # Puedes usar otros modelos como 'yolov8s.pt', 'yolov8m.pt', etc.

# Set training configuration
config = {
    'imgsz': 480,
    'batch': 32,
    'epochs': args.epochs,
    'data': args.input_path,  # Your custom dataset configuration in YOLO format
    'project': args.output_path,
    'name': args.exp_name,
    'device': 0,  # Use GPU if available
}

if args.mode == "train":
    model.train(**config) # Train model and save, also stores validation results
elif args.mode == "val":
    model = YOLO(f'{args.input_path}/weights/best.pt')  # Replace with the path to your trained model

    # Validate the model on your dataset
    results = model.val(data='conf/nuScenes_data.yaml',project= args.output_path, name="validation_data")  # Replace with the path to your dataset config

    # Get class names from the dataset
    class_names = results.names
    # Get per-class AP values
    # `metrics` is a dictionary containing various evaluation metrics, including per-class mAP
    ap50 = results.box.ap50
    ap5095 = results.box.ap

    final_results = {}

    # Get per-class results
    for i, class_name in class_names.items():
        class_result = results.box.class_result(i)
        print(f"Results for class {class_name}: {results.box.class_result(i)}")
        final_results[class_name] = {}
        final_results[class_name]["precision"] = class_result[0]
        final_results[class_name]["recall"] = class_result[1]
        final_results[class_name]["ap50"] = class_result[2]
        final_results[class_name]["ap5095"] = class_result[3]
        final_results["map50"] = results.box.map50
        final_results["map5090"] = results.box.map
        final_results["precision"] = results.box.mp
        final_results["recall"] = results.box.mr
    with open(f"{args.output_path}/validation_data/final_results.json", "w") as f:
        json.dump(final_results, f)

elif args.mode == "predict":
    results = []
    model = YOLO(f'{args.input_path}/weights/best.pt')  # Replace with the path to your trained model
    if os.path.isdir(args.img_path):
        start_time = time.time()
        images = [os.path.join(args.img_path, s) for s in os.listdir(args.img_path)]
        results = model.predict(images)
        end_time = time.time()
    else:
        results = model.predict(args.img_path)
        results[0].show()