import cv2 
import numpy as np 
import onnxruntime
import os

class FastReID:
  """
  FastReID: Fast Re-ID model wrapper for computing appearance features from images.
  
  This class wraps an ONNX-format Re-Identification (ReID) model that computes deep learning-based
  appearance features (embeddings) from images. These features capture the visual characteristics
  of objects (persons or vehicles) and can be used to compare similarity between different
  detections across video frames.
  
  Supported models:
  - Person ReID: Input size 128x256, optimized for person appearance features
  - Vehicle ReID: Input size 256x256, optimized for vehicle appearance features
  
  The model:
  1. Takes cropped images as input (BGR format from OpenCV)
  2. Preprocesses them (convert to RGB, resize, normalize)
  3. Runs inference on GPU using ONNX Runtime
  4. Returns L2-normalized feature vectors suitable for distance metrics (cosine, Euclidean)
  
  Typical usage:
  - Create one FastReID instance for persons and one for vehicles
  - Extract appearance vectors for detections to compute similarity scores
  - Use cosine distance between vectors to measure object similarity
  """
  def __init__(self, onnx_path, reid_type) -> None:
    """
    Initialize FastReID model wrapper.
    
    Args:
        onnx_path (str): Path to the ONNX model file. The model should be a pre-trained
                        ReID model exported to ONNX format.
        reid_type (str): Type of ReID model, either:
                        - 'person': For person re-identification (128x256 input size)
                        - 'vehicle': For vehicle re-identification (256x256 input size)
    
    Raises:
        ValueError: If reid_type is neither 'person' nor 'vehicle'
        FileNotFoundError: If the ONNX model file at onnx_path does not exist
        RuntimeError: If ONNX Runtime cannot initialize the model or if CUDA is not available
    
    Note:
        - Requires NVIDIA GPU with CUDA support for optimal performance
        - Falls back to CPU execution if GPU is unavailable
    """
    self.ort_sess = onnxruntime.InferenceSession(onnx_path,
    providers=['CUDAExecutionProvider']
  )
    self.reid_type = reid_type # person or vehicle

    if self.reid_type == 'person':
      self.image_height = 256
      self.image_width = 128
    elif self.reid_type == 'vehicle':
      self.image_height = 256
      self.image_width = 256
    else:
      raise ValueError("Invalid reid_type. Must be 'person' or 'vehicle'.")


  def preprocess(self, image_array):
    """
    Preprocess image for model inference.
    
    Steps performed:
    1. Convert BGR to RGB (OpenCV reads images in BGR format, but models expect RGB)
    2. Resize to model input dimensions (128x256 for person, 256x256 for vehicle)
    3. Convert to float32 and normalize to [0, 1] range
    4. Transpose from HxWxC to CxHxW format
    5. Add batch dimension to create (1, 3, H, W) tensor for batch inference
    
    Args:
        image_array (np.ndarray): Input image in BGR format (as read by cv2.imread or cv2.VideoCapture).
                                 Shape should be (height, width, 3).
    
    Returns:
        np.ndarray: Preprocessed image tensor ready for model inference.
                   Shape: (1, 3, self.image_height, self.image_width)
                   Type: float32
                   Range: [0, 1] (normalized)
    
    Note:
        - Input image should be a cropped region containing the object of interest
        - Aspect ratio may be distorted due to resizing to fixed model input dimensions
    """
    # The input image_array is expected to be BGR from cv2.VideoCapture or similar
    original_image = image_array

    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img


  def normalize(self, nparray, order=2, axis=-1):
    """
    Normalize a N-D numpy array using L-norm along the specified axis.
    
    This function applies L-norm normalization to make feature vectors unit length,
    which is important for distance metrics like cosine similarity. The normalization
    ensures that the distance between two vectors depends only on their direction,
    not their magnitude.
    
    Args:
        nparray (np.ndarray): Input array to normalize. Can be any shape.
        order (int): Order of the norm to compute (default: 2)
                    - order=2: L2 norm (Euclidean distance) - most common for feature normalization
                    - order=1: L1 norm (Manhattan distance)
        axis (int): Axis along which to compute the norm (default: -1)
                   - axis=-1: Normalize along the last axis (typically the feature dimension)
                   - Other axes can be used for batch normalization
    
    Returns:
        np.ndarray: Normalized array with same shape as input.
                   All vectors will have unit L-norm along the specified axis.
    
    Note:
        - Adds small epsilon (np.finfo(np.float32).eps ≈ 1e-7) to avoid division by zero
        - Preserves dimensions using keepdims=True
        - For feature vectors shape (1, 256), normalizes each feature dimension
    
    Example:
        >>> feat = np.array([[3.0, 4.0]])  # L2 norm = 5.0
        >>> normalized = normalize(feat)
        >>> print(normalized)  # Output: [[0.6, 0.8]] with L2 norm = 1.0
    """
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)

  def run_inference_on_frame(self, frame_array):
    """
    Extract appearance feature vector from an image using the ReID model.
    
    This is the main method for computing deep learning-based appearance features.
    The complete pipeline is:
    1. Preprocess the input image (color conversion, resizing, normalization)
    2. Run ONNX model inference on GPU
    3. Normalize the output feature vector to unit L2 norm
    
    Args:
        frame_array (np.ndarray): Input image in BGR format (as from cv2.imread or video frame).
                                 Should be a cropped region containing the object of interest.
                                 Shape: (height, width, 3)
    
    Returns:
        np.ndarray: Normalized feature vector (embedding) from the ReID model.
                   Shape: (1, feature_dim) where feature_dim is typically 256 or 512
                   Type: float32
                   Norm: L2-normalized (unit vector)
    
    Note:
        - The input image should be a tight crop around the target object (person or vehicle)
        - Feature vectors can be compared using cosine similarity or Euclidean distance
        - Cosine similarity is recommended: sim = np.dot(feat1, feat2.T) / (||feat1|| * ||feat2||)
        - Since features are L2-normalized, cosine(feat1, feat2) = np.dot(feat1, feat2.T)
    
    Raises:
        RuntimeError: If ONNX Runtime inference fails (e.g., GPU memory issues)
    
    Example:
        >>> from scipy.spatial.distance import cosine
        >>> feat1 = reid_model.run_inference_on_frame(crop1)
        >>> feat2 = reid_model.run_inference_on_frame(crop2)
        >>> similarity = 1 - cosine(feat1.flatten(), feat2.flatten())
        >>> print(f"Similarity: {similarity}")  # Higher values mean more similar
    """
    # Call the updated preprocess method
    img = self.preprocess(frame_array)

    # Run inference
    feat = self.ort_sess.run(None, {self.ort_sess.get_inputs()[0].name: img})[0]
    # Normalize feature vector
    feat = self.normalize(feat)
    return feat