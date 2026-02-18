import cv2 
import numpy as np
from fastreid.reid import FastReID
import os


def extract_appearance_vector_from_frame(video_path:str, frame_id:int, reid_model: FastReID, bounding_box: list):
    # Assuming frame_array is a numpy array representing the frame
    # and reid_model is a FastReID model instance
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    else:
        cap = cv2.VideoCapture(video_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame_array = cap.read()

        if ret:
            # Crop the frame to the bounding box
            x, y, w, h = bounding_box
            x2 = w + x
            y2 = h + y
            frame_array = frame_array[int(y):int(y2), int(x):int(x2)]
            
            appearance_vector = reid_model.run_inference_on_frame(frame_array)

            cap.release()
            return appearance_vector
            
        else:
            cap.release()
            raise ValueError(f"Could not read frame {frame_id} from video {video_path}")


def calculate_iou(box1_left, box1_top, box1_width, box1_height, box2_left, box2_top, box2_width, box2_height):
    # Convert (left, top, width, height) to (x1, y1, x2, y2) format
    box1 = [box1_left, box1_top, box1_left + box1_width, box1_top + box1_height]
    box2 = [box2_left, box2_top, box2_left + box2_width, box2_top + box2_height]

    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of union
    union_area = float(box1_area + box2_area - intersection_area)

    # Handle case where union_area is zero to avoid division by zero
    if union_area == 0:
        return 0.0

    # Compute the IoU
    iou = intersection_area / union_area

    return iou
