import cv2 
import numpy as np
from fastreid.reid import FastReID
import os
import math
from scipy.spatial.distance import euclidean


def extract_appearance_vector_from_frame(video_path:str, frame_id:int, reid_model: FastReID, bounding_box: list):
    """
    Extract appearance feature vector for a region of interest from a specific video frame.
    
    This function uses a pre-trained ReID (Re-Identification) model to compute deep appearance
    features for a cropped region of a video frame. These features can be used to compare the
    visual similarity between different detections across frames.
    
    Args:
        video_path (str): Path to the video file to extract frame from
        frame_id (int): Frame number to extract (1-based indexing, where frame 1 is the first frame).
                       Will be converted to 0-based indexing internally.
        reid_model (FastReID): Pre-trained ReID model instance (e.g., from FastReID class).
                              Must have a run_inference_on_frame() method.
        bounding_box (list): Bounding box in format [x, y, width, height] where:
                            - x: left coordinate of the box
                            - y: top coordinate of the box
                            - width: width of the box
                            - height: height of the box
    
    Returns:
        np.ndarray: Feature vector (embedding) computed by the ReID model for the cropped region.
                   Shape is typically (1, 256) or similar depending on the model architecture.
    
    Raises:
        FileNotFoundError: If the video file does not exist at video_path
        ValueError: If the specified frame_id cannot be read from the video
    
    Example:
        >>> features = extract_appearance_vector_from_frame(
        ...     video_path='video.mp4',
        ...     frame_id=100,
        ...     reid_model=person_reid_model,
        ...     bounding_box=[50, 100, 200, 300]
        ... )
    """
    frame_id = frame_id - 1
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    else:
        cap = cv2.VideoCapture(video_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES-1, frame_id)
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


def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    IoU is a metric used to measure the overlap between two bounding boxes. It is computed as:
    IoU = Area_Intersection / Area_Union
    
    This metric ranges from 0 (no overlap) to 1 (perfect overlap) and is commonly used for:
    - Evaluating detection quality
    - Matching detections across frames
    - Non-maximum suppression (NMS)
    
    Args:
        bbox1 (list): First bounding box in format [x, y, width, height] where:
                     - x: left coordinate
                     - y: top coordinate
                     - width: box width
                     - height: box height
        bbox2 (list): Second bounding box in the same format as bbox1
    
    Returns:
        float: IoU value between 0 and 1, where:
              - 0 means no overlap
              - 1 means complete overlap
              - 0.5 means 50% overlap
    
    Example:
        >>> iou = calculate_iou([0, 0, 100, 100], [50, 50, 100, 100])
        >>> print(iou)  # Output: 0.14285714...
    """
    # Convert (left, top, width, height) to (x1, y1, x2, y2) format
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2


    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]

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

def get_bounding_box_ratio(bbox1:list, bbox2:list):
    """
    Compute the aspect ratio between two bounding boxes.
    
    This function calculates the ratio of widths and heights between two bounding boxes.
    These ratios are useful for determining if detections are likely to be the same object.
    For example, if width_ratio and height_ratio are both close to 1.0, the objects have
    similar sizes, which is a good indicator they might be the same person/vehicle.
    
    Args:
        bbox1 (list): First bounding box in format [x, y, width, height]
        bbox2 (list): Second bounding box in format [x, y, width, height]
    
    Returns:
        list: A 2-element list [width_ratio, height_ratio] where:
             - width_ratio = bbox1_width / bbox2_width
             - height_ratio = bbox1_height / bbox2_height
             Ratios close to 1.0 indicate similar sizes
    
    Example:
        >>> ratio = get_bounding_box_ratio([0, 0, 100, 200], [0, 0, 50, 100])
        >>> print(ratio)  # Output: [2.0, 2.0]
    """
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2

    return [w1/w2, h1/h2]


def get_direction(bbox1:list, bbox2:list):
    """
    Compute the directional angle from one bounding box to another.
    
    This function calculates the angle (in radians) of movement from the first bounding box
    to the second. The direction can indicate whether an object is moving left, right, up, down,
    or in any diagonal direction. This is useful for predicting object motion patterns and
    validating trajectory continuity.
    
    Args:
        bbox1 (list): First bounding box in format [x, y, width, height] (starting position)
        bbox2 (list): Second bounding box in format [x, y, width, height] (ending position)
    
    Returns:
        float: Direction angle in radians, computed as atan2(dx, dy) where:
              - dx = bbox2_x - bbox1_x (horizontal displacement)
              - dy = bbox2_y - bbox1_y (vertical displacement)
              - Range: [-π, π]
              - 0 radians = moving down
              - π/2 radians = moving right
              - -π/2 radians = moving left
    
    Note:
        The function uses the top-left corner coordinates (x, y) of each bounding box.
        The direction is relative to the box's position, not its center.
    
    Example:
        >>> direction = get_direction([0, 0, 100, 100], [100, 100, 100, 100])
        >>> print(direction)  # Output: angle in radians
    """
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2
    
    dx = x2 - x1
    dy = y2 - y1
    direction = math.atan2(dx,dy)

    return direction 

def get_euclidean_distance(bbox1:list, bbox2:list):
    """
    Calculate the Euclidean distance between the top-left corners of two bounding boxes.
    
    This function computes the straight-line distance between the initial coordinates (x, y)
    of two bounding boxes. A smaller distance indicates the two detections are spatially close,
    which is a good indicator they might represent the same object at different times.
    
    Args:
        bbox1 (list): First bounding box in format [x, y, width, height]
        bbox2 (list): Second bounding box in format [x, y, width, height]
    
    Returns:
        float: Euclidean distance between (x1, y1) and (x2, y2):
              distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
              Units are in pixels. Higher values indicate larger spatial separation.
    
    Note:
        Unlike get_direction() which gives angle, this function gives magnitude of displacement.
        Uses only the top-left corner coordinates (x, y) of the bounding boxes, not centers.
    
    Example:
        >>> distance = get_euclidean_distance([0, 0, 100, 100], [30, 40, 100, 100])
        >>> print(distance)  # Output: 50.0 (sqrt(30^2 + 40^2))
    """
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2

    return euclidean([x1,y1],[x2,y2])
    