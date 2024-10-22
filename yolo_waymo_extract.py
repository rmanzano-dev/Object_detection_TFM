import os

import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset

TYPE_UNKNOWN = 0
TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST = 4

# Classes for detection: 'vehicle', 'pedestrian', 'cyclist', 'traffic_sign'
CLASS_MAPPING = {
    TYPE_VEHICLE: 0,
    TYPE_PEDESTRIAN: 1,
    TYPE_CYCLIST: 2
}

def extract_tfrecord(tfrecord_path, output_image_dir, output_label_dir, segment):
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    dataset_len = sum(1 for _ in dataset)
    for idx, data in tqdm(enumerate(dataset), total=dataset_len):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # Extract images from the frame
        for image in frame.images:
            img_name = f"segment_{segment}_frame_{idx}_camera_{image.name}.jpg"
            img_path = os.path.join(output_image_dir, img_name)
            # Save image
            with open(img_path, 'wb') as f:
                f.write(image.image)
            # Extract the corresponding camera calibration data to get image width and height
            for camera_calibration in frame.context.camera_calibrations:
                if camera_calibration.name == image.name:
                    img_width = camera_calibration.width
                    img_height = camera_calibration.height
                    break
            # Extract labels and annotations (bounding boxes) from camera labels
            for camera_labels in frame.camera_labels:
                if camera_labels.name == image.name:  # Match the camera to the image
                    label_file_path = os.path.join(output_label_dir, f"segment_{segment}_frame_{idx}_camera_{image.name}.txt")
                    with open(label_file_path, 'w') as label_file:
                        for label in camera_labels.labels:
                            # Class label and bounding boxes
                            class_id = CLASS_MAPPING.get(label.type, -1)  # Get class ID
                            if class_id == -1:
                                continue  # Skip unknown class
                            # Extract bounding box
                            box = label.box
                            x_center = (box.center_x) / img_width
                            y_center = (box.center_y) / img_height
                            width = box.length / img_width
                            height = box.width / img_height

                            # Write in YOLO format: <class_id> <x_center> <y_center> <width> <height>
                            label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def convert_tfrecords_to_yolo(tfrecord_dir, output_image_dir, output_label_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    counter = 0
    # Process all TFRecord files in the directory
    for tfrecord_file in tqdm(os.listdir(tfrecord_dir)):
        
        if tfrecord_file.endswith('.tfrecord'):
            tfrecord_path = os.path.join(tfrecord_dir, tfrecord_file)
            extract_tfrecord(tfrecord_path, output_image_dir, output_label_dir, counter)
            counter = counter + 1

# Define paths for the input TFRecords and output image/label directories
tfrecord_dir = 'TFM_Data/validation'  # Directory containing TFRecord files
output_image_dir = 'YOLO_DATA/validation/images'  # Directory to save extracted images
output_label_dir = 'YOLO_DATA/validation/labels'  # Directory to save YOLO annotations

convert_tfrecords_to_yolo(tfrecord_dir, output_image_dir, output_label_dir)
