import json
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from argparser import get_args
from metrics import (get_metrics, get_precision_recall,
                     get_precision_recall_per_class,
                     plot_precision_recall_curve)
from utils import draw_boxes_opencv

classes = ["VEHÍCULO", "PEATÓN", "CICLISTA"]

# Load YOLOv8 annotations
def load_yolov8_annotation(annotation_file, img_width, img_height):
    boxes = []
    labels = []
    
    with open(annotation_file, 'r') as f:
        for line in f:
            # Parse YOLOv8 format: <class_id> <x_center> <y_center> <width> <height>
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            # Convert YOLO normalized format to absolute pixel values
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            # Convert (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2

            # Append to boxes and labels list
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(int(class_id) + 1)  # Increment by 1 to account for background class (class_id 0 -> label 1)
    
    # Convert to tensors
    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)
    
    return boxes, labels

# Custom Dataset Class for Faster R-CNN using YOLOv8 Annotations
class YoloToFasterRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size
        
        # Load corresponding YOLOv8 annotation
        annotation_file = os.path.join(self.annotations_dir, self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        boxes, labels = load_yolov8_annotation(annotation_file, img_width, img_height)
        
        # Convert the image to a tensor
        img = torchvision.transforms.functional.to_tensor(img)

        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        # Skip images with no annotations
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_faster_rcnn_model(num_classes, pt=False):
    # Load a pre-trained Faster R-CNN model from torchvision (pretrained on COCO)
    if pt:
        print("Using PRE-TRAINED model")
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    else:
        print("Using SCRATCH model")
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn()
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one for Waymo (num_classes = 4 including background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# Training function (extended to include validation)
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if torch.isnan(losses):
            print("NaN detected in losses")
            break

        # Zero the gradients
        optimizer.zero_grad()
        # Backward pass and optimize
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()

    return total_loss / len(data_loader)


# Validation function
def validate_one_epoch(model, data_loader, device, num_classes):
    model.eval()  # Set model to evaluation mode
    # Run inference on the dataset
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # Disable gradient computation for validation
        for images, targets in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Forward pass
            outputs = model(images, targets)

            all_predictions.append(outputs[0])
            all_targets.append(targets[0])
            # print(loss_dict[0].keys())
            # losses = sum(loss for loss in loss_dict.values())

            # total_val_loss += losses.item()

    metrics, precision_tensor  = get_metrics(all_predictions, all_targets)
    precision_per_class, recall_per_class = get_precision_recall_per_class(all_predictions, all_targets)

    keep = ["map", "map_50", "map_per_class", "classes"]
    final_metrics = {k: metrics[k] for k in keep}

    for idx, class_name in enumerate(classes):
        final_metrics[class_name] = {}
        final_metrics[class_name]["ap50"] = torch.mean(precision_tensor[0,:, idx, 0, 2]).cpu().tolist()
        final_metrics[class_name]["precision"] = precision_per_class[idx+1]
        final_metrics[class_name]["recall"] = recall_per_class[idx+1]
        final_metrics[class_name]["ap5090"] = metrics["map_per_class"][idx].cpu().tolist()
    final_metrics["precision"] = np.mean(list(precision_per_class.values()))
    final_metrics["recall"] = np.mean(list(recall_per_class.values()))
    for key, value in final_metrics.items():
        if type(value) == torch.Tensor:
            final_metrics[key] = value.cpu().tolist()

    with open(f"FasterRCNN_Metrics_Waymo.json", "w") as outfile: 
        json.dump(final_metrics, outfile)
    #plot_precision_recall_curve(precision_tensor, num_classes)
    return final_metrics 


def preprocess_image(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Define the image transforms
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert PIL Image to Tensor
    ])
    
    # Apply the transforms to the image
    img = transform(img)
    
    # Add batch dimension (model expects [N, C, H, W])
    return img.unsqueeze(0)
# Function to train with validation
def train_faster_rcnn_with_validation(model, train_loader, val_loader, num_epochs, device, num_classes):
    model.to(device)
    
    # Define the optimizer (use SGD with momentum)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validation
        metrics = validate_one_epoch(model, val_loader, device, num_classes=num_classes)

        with open(f"{output_path}/fasterRCNN_ep-{epoch + 1}_json", "w") as outfile: 
            json.dump(metrics, outfile)

        torch.save(model.state_dict(), f"{output_path}/fasterRCNN_ep-{epoch + 1}_best.pth")

    print("Training complete!")

args = get_args()
output_path = args.output_path

# Load the Faster R-CNN model with the number of classes (3 classes + background)
num_classes = 4  # 3 classes (vehicle, pedestrian, cyclist) + 1 for background
model = get_faster_rcnn_model(num_classes, pt=args.pretrained)

# Train the model with validation
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_epochs = args.epochs

if args.mode == "val":
    images_dir_val = f'{args.input_path}/validation/images'
    annotations_dir_val = f'{args.input_path}/validation/labels'
    val_dataset = YoloToFasterRCNNDataset(images_dir_val, annotations_dir_val)
    val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn)
    model.load_state_dict(torch.load("Waymo_FasterRCNNFinal_resultsT/fasterRCNN_ep-4_best.pth"))

    model.to(device)
    # class_names = ['__background__', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST']
    metrics = validate_one_epoch(model, val_loader, device, num_classes)

elif args.mode == "predict":
    model.load_state_dict(torch.load("Waymo_FasterRCNNFinal_results/fasterRCNN_ep-4_best.pth"))
    image_path = args.img_path

    import time
    start_time = time.time()
    img = preprocess_image(image_path)

    # Move the image and model to the same device (CPU or GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    model.to(device)
    img = img.to(device)

    with torch.no_grad():
        prediction = model(img)

    end_time = time.time()
    elapsed_time = end_time - start_time
    class_names = ['__background__', 'VEHICULO', 'PEATON', 'CICLISTA']
    draw_boxes_opencv(image_path, args.output_path, prediction[0]["boxes"], prediction[0]["labels"], prediction[0]["scores"], threshold=0.5, class_names=class_names)
    print("Time taken for prediction " + str(elapsed_time))
else:
    if args.pretrained:
        output_path = output_path + "_PT"

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=False)
    # Example usage: Train and validation datasets
    images_dir_train = f'{args.input_path}/train/images'
    annotations_dir_train = f'{args.input_path}/train/labels'
    images_dir_val = f'{args.input_path}/validation/images'
    annotations_dir_val = f'{args.input_path}/validation/labels'

    # Create the train and validation datasets
    train_dataset = YoloToFasterRCNNDataset(images_dir_train, annotations_dir_train)
    val_dataset = YoloToFasterRCNNDataset(images_dir_val, annotations_dir_val)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn)
    train_faster_rcnn_with_validation(model, train_loader, val_loader, num_epochs, device, num_classes=num_classes)
