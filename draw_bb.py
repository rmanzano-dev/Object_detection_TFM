import cv2

# Class names as per the CLASS_MAPPING used earlier
CLASS_NAMES = ['vehicle', 'pedestrian', 'cyclist']
def display_image_with_labels(image_path, label_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Get the image dimensions
    img_height, img_width = image.shape[:2]

    # Read the label file (YOLO format)
    with open(label_path, 'r') as label_file:
        labels = label_file.readlines()

    # Loop through each label and draw the bounding box
    for label in labels:
        # Each line in label file is formatted as: <class_id> <x_center> <y_center> <width> <height>
        class_id, x_center, y_center, width, height = map(float, label.strip().split())

        # Convert normalized coordinates back to original image scale
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Calculate the top-left and bottom-right coordinates of the bounding box
        x_min = int(x_center - (width / 2))
        y_min = int(y_center - (height / 2))
        x_max = int(x_center + (width / 2))
        y_max = int(y_center + (height / 2))

        # Draw the bounding box on the image
        color = (0, 255, 0)  # Green box for bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Put the class name above the bounding box
        label_text = CLASS_NAMES[int(class_id)]
        cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the image with bounding boxes
    cv2.imshow('Image with Labels', image)
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()

# Example usage
image_path = 'NuScenes_YOLO/images/0a0d6b8c2e884134a3b48df43d54c36a.jpg'
label_path = 'NuScenes_YOLO/labels/0a0d6b8c2e884134a3b48df43d54c36a.txt'

display_image_with_labels(image_path, label_path)
