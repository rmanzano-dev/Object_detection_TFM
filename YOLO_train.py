from ultralytics import YOLO

from argparser import get_args

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

# Train model and save, also stores validation results
model.train(**config)