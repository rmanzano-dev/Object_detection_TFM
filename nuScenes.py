import os

from nuscenes.nuscenes import NuScenes
from PIL import Image
from tqdm import tqdm

from get_2d_annotations_nuscenes import get_2d_boxes

# Define path where the NuScenes dataset has been downloaded
dataroot = 'NuScenes'
nusc = NuScenes(version='v1.0-mini', dataroot="NuScenes", verbose=True)
# Output path for NuScenes data extracted
output_dir = 'NuScenes_YOLO'
os.makedirs(output_dir, exist_ok=True)

# Definir las clases de interés
target_classes = ['vehicle.car', 'human.pedestrian.adult', 'vehicle.bicycle']
cats = nusc.category

class_mapping = {'vehicle.car': 0, 'human.pedestrian.adult': 1, 'vehicle.bicycle': 2}


def convert_bbox_to_yolo(img_width, img_height, bbox):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return [x_center, y_center, width, height]


def save_yolo_annotations(sample_data_token, output_file, image_output):
    # Get annotations for the sample
    sample_data = nusc.get('sample_data', sample_data_token)
    camera_filename = sample_data['filename']
    img_path = os.path.join(nusc.dataroot, camera_filename)
    sample_annotations = []
    # Load the image to get dimensions and save
    img = Image.open(img_path)
    img_width, img_height = img.size

    # Call function extracted from a NuScenes script, to get 2d projection from 3d bounding boxes
    boxes = get_2d_boxes(sample_data_token, nusc)
    with open(output_file, 'w') as f:
        for annotation in boxes:
            # Filter only desired classes
            if annotation["category_name"] not in target_classes:
                continue

            category_id = class_mapping[annotation["category_name"]]

            # Convert box corners to YOLO format
            yolo_bbox = convert_bbox_to_yolo(img_width, img_height, annotation['bbox_corners'])
            class_id = category_id
            sample_annotations.append([class_id, *yolo_bbox])

            # Save annotation and image
            if sample_annotations != []:
                img.save(image_output)
                with open(output_file, 'w') as f:
                    for ann in sample_annotations:
                        f.write(' '.join(map(str, ann)) + '\n')

# Extract annotations per sample
for sample in tqdm(nusc.sample):
    for camera in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']:
        sample_data_token = sample['data'][camera]
        output_file = os.path.join(output_dir, "labels", f"{sample['token']}.txt")
        image_output = os.path.join(output_dir, "images", f"{sample['token']}.jpg")
        save_yolo_annotations(sample_data_token, output_file, image_output)

print("Extracción de anotaciones completada.")
