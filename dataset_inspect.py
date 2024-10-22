import os

import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset

TYPE_UNKNOWN = 0
TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_SIGN = 3
TYPE_CYCLIST = 4

CLASS_MAPPING = {
    TYPE_VEHICLE: 0,
    TYPE_PEDESTRIAN: 1,
    TYPE_SIGN: 2,
    TYPE_CYCLIST: 3
}

basepath = "TFM_Datav2/train/"
files = os.listdir(basepath)
tfrecord_path = basepath + "/" + files[0]
dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
dataset_len = sum(1 for _ in dataset)
for idx, data in tqdm(enumerate(dataset), total=dataset_len):
    segment = 6
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    # Extract images from the frame
    for image in frame.images:
        img_name = f"test_img.jpg"
        with open(img_name, 'wb') as f:
            f.write(image.image)
        # Extract the corresponding camera calibration data to get image width and height
        for camera_calibration in frame.context.camera_calibrations:
            if camera_calibration.name == image.name:
                img_width = camera_calibration.width
                img_height = camera_calibration.height
                break
        # Extract labels and annotations (bounding boxes) from camera labels
        for camera_labels in frame.camera_labels:
            if camera_labels.name == image.name and image.name == 4:  # Match the camera to the image
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