import cv2

def draw_boxes_opencv(image_path, boxes, labels, scores, threshold=0.5, class_names=None):
    image = cv2.imread(image_path)
    
    for i in range(len(boxes)):
        if scores[i] >= threshold:
            box = boxes[i].cpu().numpy().astype(int)
            label = labels[i].cpu().numpy()
            score = scores[i].cpu().numpy()
            
            # Draw rectangle (bounding box)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            
            # Prepare label and score text
            class_name = class_names[label] if class_names else f'Class {label}'
            caption = f'{class_name}: {score:.2f}'
            
            # Draw label text above the bounding box
            cv2.putText(image, caption, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Show the image
    cv2.imshow('Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

