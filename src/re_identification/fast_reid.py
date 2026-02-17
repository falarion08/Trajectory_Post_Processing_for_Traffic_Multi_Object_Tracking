import cv2 
import numpy as np 
import onnxruntime
import os


class FastReID:
  def __init__(self, reid_type) -> None:

      self.reid_type = reid_type # person or vehicle

      if self.reid_type == 'person':
        self.image_height = 256
        self.image_width = 128
        onnx_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'fastreid_person.onnx')
      elif self.reid_type == 'vehicle':
        self.image_height = 256
        self.image_width = 256
        onnx_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'fastreid_vehicle.onnx')
      else:
        raise ValueError("Invalid reid_type. Must be 'person' or 'vehicle'.")
     
     
      self.ort_sess = onnxruntime.InferenceSession(onnx_path,
      providers=['CUDAExecutionProvider']
        )

  def preprocess(self, image_array):
    # The input image_array is expected to be BGR from cv2.VideoCapture or similar
    original_image = image_array

    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img


  def normalize(self, nparray, order=2, axis=-1):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)

  def run_inference_on_frame(self, frame_array):
    # Call the updated preprocess method
    img = self.preprocess(frame_array)

    # Run inference
    feat = self.ort_sess.run(None, {self.ort_sess.get_inputs()[0].name: img})[0]
    # Normalize feature vector
    feat = self.normalize(feat)
    return feat
