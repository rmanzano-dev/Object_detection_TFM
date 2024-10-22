import os

import cv2
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

# Define output of extracted data
output_dir = 'waymo_yolo_dataset/validation/'
images_dir = os.path.join(output_dir, 'images')
labels_dir = os.path.join(output_dir, 'labels')

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Path to TFRecordFiles
tfrecord_path = '/media/rafadev/Seagate Expansion Drive/TFM/TFM_Data/validation/'
files = os.listdir(tfrecord_path)
files_abs = [tfrecord_path+item for item in files]
# Loads all TfRecords
dataset = tf.data.TFRecordDataset(files_abs)

# Mapeo de clases Waymo a clases YOLO
waymo_to_yolo_class = {
    1: 0,  # VEHICLE
    2: 1,  # PEDESTRIAN
    4: 2   # CYCLIST
}

# Iterar sobre cada frame en el archivo TFRecord
for idx, data in enumerate(dataset):
    # Parsear el frame
    frame = open_dataset.Frame()
    frame.ParseFromString(data.numpy())

    # Obtener las imágenes (usaremos la cámara frontal, por ejemplo)
    for image in frame.images:
        if image.name == open_dataset.CameraName.FRONT:
            # Decodificar la imagen
            img = tf.image.decode_jpeg(image.image).numpy()
            
            # Guardar la imagen
            image_filename = f'image_{idx:06d}.jpg'
            image_path = os.path.join(images_dir, image_filename)
            cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Crear el archivo de etiquetas YOLO
            label_filename = f'image_{idx:06d}.txt'
            label_path = os.path.join(labels_dir, label_filename)

            with open(label_path, 'w') as f:
                # Obtener las anotaciones para la cámara frontal
                for camera_label in frame.camera_labels:
                    if camera_label.name == open_dataset.CameraName.FRONT:
                        for label in camera_label.labels:
                            class_id = waymo_to_yolo_class.get(label.type, None)
                            if class_id is not None:
                                # Normalizar las coordenadas al rango [0, 1]
                                img_height, img_width, _ = img.shape
                                x_center = (label.box.center_x) / img_width
                                y_center = (label.box.center_y) / img_height
                                width = label.box.length / img_width
                                height = label.box.width / img_height
                                
                                # Escribir las anotaciones al archivo
                                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
