import json
import os

import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from argparser import get_args
from metrics import calculate_map_per_class
from utils import draw_boxes_opencv


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
            all_predictions.append(outputs)
            all_targets.append(targets)
            # print(loss_dict[0].keys())
            # losses = sum(loss for loss in loss_dict.values())

            # total_val_loss += losses.item()

    precision_per_class, recall_per_class, ap_per_class = calculate_map_per_class(all_predictions, all_targets, num_classes)

    return precision_per_class, recall_per_class, ap_per_class

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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.95, weight_decay=0.0005)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validation
        precision_per_class, recall_per_class, ap_per_class = validate_one_epoch(model, val_loader, device, num_classes=num_classes)

        metrics = {}
        for k in precision_per_class.keys():
            metrics[k] = [precision_per_class[k], recall_per_class[k], ap_per_class[k]]

        with open(f"{output_path}/fasterRCNN_ep-{epoch + 1}_json", "w") as outfile: 
            json.dump(metrics, outfile)

        torch.save(model.state_dict(), f"{output_path}/fasterRCNN_ep-{epoch + 1}_best.pth")

    print("Training complete!")

args = get_args()
output_path = args.output_path

if args.pretrained:
    output_path = output_path + "_PT"
    print(output_path)
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

# Load the Faster R-CNN model with the number of classes (3 classes + background)
num_classes = 4  # 3 classes (vehicle, pedestrian, cyclist) + 1 for background
model = get_faster_rcnn_model(num_classes, pt=args.pretrained)

# Train the model with validation
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_epochs = args.epochs

if args.mode == "val":

    model.load_state_dict(torch.load("NuScenes-metrics/fasterRCNNPT_ep-15_best.pth"))

    model.to(device)
    # class_names = ['__background__', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST']
    # precision_per_class, recall_per_class, ap_per_class = validate_one_epoch(model, val_loader, device, num_classes)

elif args.mode == "predict":
    model.load_state_dict(torch.load("fasterRCNNPT_ep-15_best.pth"))
    image_path = args.img_path

    img = preprocess_image(image_path)

    # Move the image and model to the same device (CPU or GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    model.to(device)
    img = img.to(device)

    with torch.no_grad():
        prediction = model(img)


    class_names = ['__background__', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST']
    draw_boxes_opencv(image_path, prediction[0]["boxes"], prediction[0]["labels"], prediction[0]["scores"], threshold=0.1, class_names=class_names)
else:
    train_faster_rcnn_with_validation(model, train_loader, val_loader, num_epochs, device, num_classes=num_classes)
