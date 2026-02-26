import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from trajectory_improvement.tracklet import Track
from fastreid.reid import FastReID
from utils.utils import (
    extract_appearance_vector_from_frame,
    calculate_iou,
    get_euclidean_distance,
    get_bounding_box_ratio,
    get_direction
)



class TrajectoryBreakPhase:
  """
  TrajectoryBreakPhase is responsible for breaking long object trajectories into smaller tracklets
  (trajectory segments) at points where sudden motion anomalies are detected.
  
  This class uses a Kalman filter to predict expected object motion and compares actual measurements
  against predictions using the Mahalanobis distance metric. When a measurement deviates significantly
  from the prediction (exceeds threshold), the trajectory is broken at that point.
  
  This segmentation is useful for:
  - Detecting occlusions or temporary disappearances
  - Identifying sudden direction/speed changes that might indicate tracking errors
  - Creating tracklets that can later be linked together in the LinkingPhase
  
  Typical workflow:
  1. Load MOT (Multi-Object Tracking) format CSV file
  2. For each unique tracked object, extract trajectory
  3. Run Kalman filter to compute Mahalanobis distances for each frame
  4. Split trajectory into segments at break points (high Mahalanobis distance)
  5. Return list of Track objects representing the broken tracklets
  """
  
  def __init__(self,input_csv_filename:str,video_fps:int,mahalanobis_distance_thresh:float=1.9) -> None:
    """
    Initialize the TrajectoryBreakPhase with MOT data.
    
    Args:
        input_csv_filename (str): Path to the MOT format CSV file with tracking data
        video_fps (int): Frames per second of the video (used to compute measurement period for Kalman filter)
        mahalanobis_distance_thresh (float): Threshold for breaking trajectories (default: 1.9).
                                           When Mahalanobis distance exceeds this, a break is detected.
    """
    self.mot_df = pd.read_csv(input_csv_filename)
    self.video_fps = video_fps
    self.thresh = mahalanobis_distance_thresh

  def process_trajectory(self,filtered_track:pd.DataFrame,distance_list:list[int]):
    """
    Split a trajectory into tracklets at points where Mahalanobis distance exceeds threshold.
    
    Algorithm:
    1. Iterate through each measurement and its corresponding Mahalanobis distance
    2. When distance >= threshold, mark as break point
    3. Extract segment from previous break point to current break point
    4. Create a Track object for each non-empty segment
    5. Return list of all tracklets
    
    Args:
        filtered_track (pd.DataFrame): DataFrame containing all detections for a single track ID,
                                      sorted by frame_number. Must include columns: frame_number,
                                      x_center, y_center, and other detection attributes.
        distance_list (list[int]): List of Mahalanobis distances computed by Kalman filter.
                                  Length should match filtered_track rows.
    
    Returns:
        list[Track]: List of Track objects representing trajectory segments (tracklets)
    """
    filtered_track = filtered_track.reset_index(drop=True) # Ensure contiguous 0-based index and drop old index
    tracklet_list = []
    start_index_copy = 0

    # Iterate through the distances and corresponding track points.
    for i in range(len(distance_list)):

      # A break is detected if the current point's Mahalanobis distance is above threshold
      if distance_list[i] >= self.thresh:

        # slice from start_index_copy up to i (exclusive)
        tracklet_segment = filtered_track.iloc[start_index_copy:i]

        if not tracklet_segment.empty:
          tracklet_records = tracklet_segment.to_dict(orient='records')
          frame_start = tracklet_records[0]['frame_number']
          frame_end = tracklet_records[-1]['frame_number']

          tracklet = Track(
              start_frame=frame_start,
              end_frame=frame_end,
              track_list = tracklet_records)
          tracklet_list.append(tracklet)

        # The new segment starts from the current point 'i'
        start_index_copy = i

    # After the loop, add the last tracklet segment if it exists
    # This segment runs from the last 'start_index_copy' to the very end of the filtered_track
    tracklet_segment = filtered_track.iloc[start_index_copy:len(filtered_track)]

    if not tracklet_segment.empty:
      tracklet_records = tracklet_segment.to_dict(orient='records')
      frame_start = tracklet_records[0]['frame_number']
      frame_end = tracklet_records[-1]['frame_number']

      tracklet = Track(start_frame=frame_start, end_frame=frame_end,track_list = tracklet_records)
      tracklet_list.append(tracklet)

    return tracklet_list


  def create_trackelts(self):
    """
    Main method that processes all trajectories from the MOT DataFrame and breaks them into tracklets.
    
    Process:
    1. Extract all unique tracker IDs from the input MOT data
    2. For each tracker ID:
       a. Filter detections for that specific ID
       b. Extract trajectory coordinates (frame_number, x_center, y_center)
       c. Initialize Kalman filter with trajectory
       d. Compute Mahalanobis distances
       e. Break trajectory into segments at anomaly points
    3. Combine all tracklets from all tracking IDs
    
    Returns:
        list[Track]: Complete list of broken tracklets from all objects in the video
    """
    # Get all unique ID in present from the output
    track_id_list = self.mot_df['tracker_id'].unique()
    broken_track_list = []

    for track_id in track_id_list:

      filtered_track = self.mot_df[self.mot_df['tracker_id'] == track_id].sort_values(by='frame_number').copy()
      trajectory_list = filtered_track[['frame_number','x_center','y_center']].values.tolist()

      kalman_filter = KalmanFilter2D(
          measurement_period= 1/self.video_fps,
          measurement_error_std= 3,
          acceleration_std= 0.2,
          measured_trajectory_points=trajectory_list
      )

      computed_distances = kalman_filter.run_kalman_filter()

      tracklet_list = self.process_trajectory(filtered_track,computed_distances)

      if tracklet_list:
        broken_track_list.extend(tracklet_list)

    return broken_track_list


class KalmanFilter2D:
  """
  A 2D Kalman Filter implementation using a Constant Acceleration Model.
  
  This filter predicts the x,y center coordinates of a moving object trajectory and computes
  the Mahalanobis distance between predicted and actual measurements. High Mahalanobis distances
  indicate anomalous detections that deviate from expected motion patterns.
  
  State Vector (6D): [x, vx, ax, y, vy, ay]
    - x, y: position
    - vx, vy: velocity
    - ax, ay: acceleration
  
  Measurement Vector (2D): [x_measured, y_measured]
  
  The filter uses:
  - State Transition Matrix (F): Predicts next state given current state
  - Measurement Matrix (H): Maps state to observable measurements
  - Process Noise Covariance (Q): Uncertainty in the motion model
  - Measurement Covariance (R): Uncertainty in the sensor measurements
  - State Covariance (P): Uncertainty in the state estimate
  """
  def __init__(self,measurement_period, measurement_error_std, acceleration_std,measured_trajectory_points):
    """
    Initialize the 2D Kalman Filter.
    
    Args:
        measurement_period (float): Time interval between measurements (typically 1/fps)
        measurement_error_std (float): Standard deviation of measurement noise (in pixels).
                                      Controls how much we trust the measurements.
        acceleration_std (float): Standard deviation of process noise (in pixels/frame^2).
                                Represents uncertainty in constant acceleration model.
        measured_trajectory_points (list): List of [frame_number, x_center, y_center] tuples
                                          representing the object's trajectory to analyze.
    """
    self.measurement_period = measurement_period
    self.measurement_error_std = measurement_error_std
    self.acceleration_std = acceleration_std
    self.measured_trajectory_points = measured_trajectory_points
    self.starting_frame = measured_trajectory_points[0][0]
    self.ending_frame = measured_trajectory_points[-1][0]
    self.innovation_residual = np.array([])
    self.inv_innovation_covariance = np.array([])


  def initialize_matrices(self):
    """
    Initialize all matrices used in the Kalman filter equations.
    
    Sets up:
    - H (Measurement Matrix): Maps 6D state to 2D measurement space
    - F (State Transition Matrix): Predicts next state using constant acceleration model
    - Q (Process Noise Covariance): Represents model uncertainty
    - R (Measurement Covariance): Represents sensor uncertainty
    - X (Initial State Estimate): Initialized with first measurement
    - P (Initial State Covariance): High uncertainty initially
    """
    dt = self.measurement_period

    # Initialize matrix H (measurement matrix)
    # Extracts x and y positions from the 6D state vector
    self.measurement_matrix = np.array([
        [1,0,0,0,0,0],
        [0,0,0,1,0,0]
    ])
    # Create 6x6 State Transition Matrix F for constant acceleration model
    # Assumes constant acceleration; position is updated based on velocity and acceleration
    self.state_transition_matrix = np.array([
        [1,dt,0.5*dt**2,0,0,0],
        [0,1,dt,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,dt,0.5*dt**2],
        [0,0,0,0,1,dt],
        [0,0,0,0,0,1]
        ])

    # Initialize matrix Q (process noise covariance)
    # Scales with acceleration_std; higher values allow more freedom in motion
    self.process_noise_covariance = np.array([
        [dt**4,dt**3,dt**2,0,0,0],
        [dt**3,dt**2,dt,0,0,0],
        [dt**2,dt,1,0,0,0],
        [0,0,0,dt**4,dt**3,dt**2],
        [0,0,0,dt**3,dt**2,dt],
        [0,0,0,dt**2,dt,1]
    ])
    self.process_noise_covariance = self.process_noise_covariance * self.acceleration_std**2

    # Initialize matrix R (measurement covariance)
    # Higher values mean less trust in measurements
    self.measurement_covariance = np.array([
        [self.measurement_error_std**2,0],
        [0,self.measurement_error_std**2]
    ])

    # Initialize state estimate X (6D vector: [x, vx, ax, y, vy, ay])
    # Initialize with first measurement position, zero velocity and acceleration
    _,first_x,first_y = self.measured_trajectory_points[0]
    self.state_estimate = np.array([first_x,0,0,first_y,0,0])
    self.state_estimate = self.state_estimate.reshape(-1,1)

    # Initialize covariance estimate P (6x6 matrix representing state uncertainty)
    # High initial values (500) indicate we start with high uncertainty that decreases with measurements
    self.covariance_estimate = np.array([
        [500,0,0,0,0,0],
        [0,500,0,0,0,0],
        [0,0,500,0,0,0],
        [0,0,0,500,0,0],
        [0,0,0,0,500,0],
        [0,0,0,0,0,500]
    ])

  def predict_state(self):
    """
    Prediction step: Advance the state and covariance estimates to the next time step.
    
    Equations:
    - State prediction: X = F * X
    - Covariance prediction: P = F * P * F^T + Q
    
    This step propagates the current state forward in time without considering measurements.
    """
    # Predict the next state using state transition matrix
    self.state_estimate = self.state_transition_matrix @ self.state_estimate
    # Predict the next covariance and add process noise uncertainty
    self.covariance_estimate = self.state_transition_matrix @ self.covariance_estimate @ self.state_transition_matrix.T + self.process_noise_covariance

  def update_state(self, measurement):
    """
    Update step: Incorporate measurement observation into the state estimate.
    
    Equations:
    - Innovation (residual): y = Z - H * X
    - Innovation Covariance: S = H * P * H^T + R
    - Kalman Gain: K = P * H^T * S^(-1)
    - State update: X = X + K * y
    - Covariance update: P = (I - K * H) * P
    
    Args:
        measurement (tuple): Two-element tuple (x_measured, y_measured) from the detection
    """
    # Convert measurement tuple to a numpy column vector (2x1)
    measurement_vector = np.array([[measurement[0]], [measurement[1]]])

    # Compute innovation residual: difference between measured and predicted values
    self.innovation_residual = measurement_vector - self.measurement_matrix @ self.state_estimate

    # Compute inverse of innovation covariance matrix
    # S = H*P*H^T + R (total uncertainty in measurement space)
    self.inv_innovation_covariance = np.linalg.inv(self.measurement_matrix @ self.covariance_estimate @ \
                                self.measurement_matrix.T + self.measurement_covariance)


    # Compute Kalman Gain: how much to trust the measurement vs prediction
    kalman_gain = self.covariance_estimate @ self.measurement_matrix.T @ self.inv_innovation_covariance

    # Update state estimate with weighted innovation
    self.state_estimate = self.state_estimate + kalman_gain @ self.innovation_residual

    # Update covariance estimate: reduce uncertainty after incorporating measurement
    self.covariance_estimate = (np.eye(self.covariance_estimate.shape[0]) - kalman_gain @ self.measurement_matrix) @ self.covariance_estimate

  def run_kalman_filter(self):
    """
    Main method: Run Kalman filter on the trajectory and compute Mahalanobis distances.
    
    Algorithm:
    1. Initialize filter matrices
    2. For each frame from start to end:
       a. Perform prediction step (advance state forward)
       b. If measurement exists in this frame:
          - Perform update step (incorporate measurement)
          - Compute Mahalanobis distance between predicted and measured positions
          - Store distance for later breakpoint detection
       c. Move to next predicted frame
    
    The Mahalanobis distance quantifies how many standard deviations away the measurement is
    from the prediction. High values indicate anomalies (potential occlusions, tracking errors, etc.)
    
    Returns:
        list[float]: List of Mahalanobis distances for each measurement point.
                    Distance = sqrt(innovation^T * S^(-1) * innovation)
                    where innovation = measurement - prediction
                    and S is the innovation covariance matrix
    """
    computed_distances = []

    i = 0
    current_frame = self.starting_frame

    self.initialize_matrices()

    # Run the kalman filter for the range of frame that contains the measurements for the unique track
    while current_frame <= self.ending_frame:
      # Prediction step: advance state forward in time
      self.predict_state()

      # Get measurement for current frame (if available)
      frame_number,x_center,y_center = self.measured_trajectory_points[i]

      if current_frame == frame_number:
        # Update step: incorporate measurement into state estimate
        self.update_state(measurement=(x_center,y_center))

        # Compute Mahalanobis distance: how many standard deviations away is the measurement?
        mahalanobis_distance = np.sqrt(self.innovation_residual.T @ self.inv_innovation_covariance @ self.innovation_residual)
        mahalanobis_distance = mahalanobis_distance[0][0] # Extract the scalar value
        mahalanobis_distance = round(mahalanobis_distance,4) # Round to 4 decimal places
        computed_distances.append(mahalanobis_distance)

        i = i + 1

      # Move to next frame
      current_frame = current_frame + 1

    return computed_distances



class LinkingPhase:
  """
  LinkingPhase is responsible for linking broken trajectories together using appearance features and spatial reasoning.
  
  This class takes a list of trajectory fragments (tracklets) and attempts to reconnect them based on:
  - Bounding box IoU and spatial proximity
  - Appearance similarity using ReID (Re-Identification) models
  - Temporal constraints (how long a tracklet can remain unmatched)
  - A pre-trained logistic regression model that predicts the likelihood of two tracklets belonging to the same object
  
  The linking process:
  1. Iterates through each frame
  2. Identifies new detections in the current frame
  3. Matches them against unmatched tracklets from previous frames
  4. Uses Hungarian algorithm for optimal assignment
  5. Combines matched tracklets with interpolated bounding boxes for frames in between
  6. Produces a final CSV output with linked trajectories
  """

  def __init__(self, video_path:str, log_reg_model:LogisticRegression, track_list:list[Track],
               csv_filename:str, vehicle_reid_path:str,person_reid_path:str,
               video_fps:int=30, lost_track_tresh:int=1, positive_match_thresh:float=0.85) -> None:
    """
    Initialize the LinkingPhase with necessary components for trajectory linking.
    
    Args:
        video_path (str): Path to the input video file
        log_reg_model (LogisticRegression): Pre-trained logistic regression model for link score prediction
        track_list (list[Track]): List of Track objects representing broken tracklets to be linked
        csv_filename (str): Output CSV filename for the linked trajectories
        vehicle_reid_path (str): Path to the ONNX model for vehicle re-identification
        person_reid_path (str): Path to the ONNX model for person re-identification
        video_fps (int): Frames per second of the video (default: 30)
        lost_track_tresh (int): Maximum time in seconds a tracklet can remain unmatched before being finalized (default: 1)
        positive_match_thresh (float): Confidence threshold [0-1] for linking two tracklets (default: 0.85)
    
    Raises:
        Exception: If the logistic regression model is not loaded correctly or is not a LogisticRegression instance
        Exception: If the track_list is empty
    """

    self.model = log_reg_model
    self.track_list = track_list.copy()
    self.video_fps = video_fps
    self.linked_track_list = []  # Store final tracks
    self.untracked_tracklets = [] # Store unmatched tracklets for each frame iteration
    self.lost_track_tresh = lost_track_tresh * self.video_fps # Maximum number of seconds a track can remain unmatched
    self.video_path = video_path
    self.positive_match_thresh = positive_match_thresh
    self.csv_filename = csv_filename
    self.person_reid = FastReID(onnx_path=person_reid_path,reid_type='person')
    self.vehicle_reid = FastReID(onnx_path=vehicle_reid_path,reid_type='vehicle')


    if not self.model or not isinstance(self.model,LogisticRegression):
      raise Exception("Model not loaded correctly. Check the model path or model type")

    if len(self.track_list) == 0:
      print("No items to process")
      return

    # Sort the list of tracks in reverse using the frame the trajectory first appeared for faster list operation
    self.track_list.sort(key=lambda x: x.start_frame,reverse=True)
    self.start_frame = self.track_list[-1].start_frame
    self.end_frame = self.track_list[0].end_frame

  def link_broken_trajectories(self):
    # 1. Examine each frame for new detections
    # 2. For each new detections in frame i, compare it with the unconnected trajectory. if no trajectory is present add it to the list
    # 3. Perform linkscore matching  and hungarian algorithm for optimal assignment
    # 4. No assignment, add it to the list
    # Interpolate


    frame_number = self.start_frame

    while frame_number <= self.end_frame:
      print(f'Processing frame: {frame_number}. Completion frame to {self.end_frame}.')

      # Update: Mark old unmatched tracklets as finalized if they exceed the lost_track_threshold
      self.update_untracked_tracklets(frame_number)

      # Check for new tracklets entering in the current frame
      detected_tracklets_from_frame = self.check_new_detections_from_frame(frame_number)

      if len(detected_tracklets_from_frame) > 0:

        if len(self.untracked_tracklets) > 0:
          # Try to link new detections with existing unmatched tracklets
          self.stitch_broken_trajectories(frame_number,detected_tracklets_from_frame)

        else:
          # If there are no trajectories to match the new detections append each item to untracked_tracklets
          self.untracked_tracklets.extend(detected_tracklets_from_frame)

      frame_number = frame_number + 1

    self.linked_track_list.extend(self.untracked_tracklets)
    self.finalize_tracklist(
        tracks= self.linked_track_list,
        csv_filename= self.csv_filename
        )

  def check_new_detections_from_frame(self,frame_ref:int):
    """
    Extract all tracklets that start in the given frame.
    
    Since track_list is sorted in reverse by start_frame, we can efficiently pop tracklets
    that begin in the current frame.
    
    Args:
        frame_ref (int): The frame number to search for new detections
    
    Returns:
        list[Track]: List of tracklets that start in frame_ref
    """
    detected_tracklets = [] # Stores current detections in reference frame

    # Keep removing the last item from the track_list as long as the start frame of the item matches the current frame

    while len(self.track_list) > 0 and self.track_list[-1].start_frame == frame_ref:
      detected_tracklets.append(self.track_list.pop())

    return detected_tracklets

  def update_untracked_tracklets(self,current_frame:int):
    """
    Mark unmatched tracklets as finalized if they exceed the lost_track_threshold.
    
    This method checks each unmatched tracklet to see if it has been unmatched for too long.
    If a tracklet's last frame is older than lost_track_tresh, it is considered finalized
    and moved to the linked_track_list. Otherwise, it remains in unmatched state.
    
    Args:
        current_frame (int): The current frame being processed
    """
    linked_trajectories = [] # Store the final trajectory
    updated_untracked_tracklets = [] # Store the new tracklets

    # Iterate each item in untracked_tracklets list to see if it falls within the number of seconds the untracked tracklet remains to be unmatched
    if len(self.untracked_tracklets) > 0:
      for tracklet in self.untracked_tracklets:

        if current_frame - tracklet.end_frame > self.lost_track_tresh:
          linked_trajectories.append(tracklet) # Tracklet considered unmatched for n number of seconds
        else:
          updated_untracked_tracklets.append(tracklet)

    self.linked_track_list.extend(linked_trajectories) # Update final track list with new considered final track list
    self.untracked_tracklets = updated_untracked_tracklets # Update untracked_tracklets are still within the window frame

  def stitch_broken_trajectories(self,current_frame,detected_tracklets:list[Track]):
    """
    Attempt to link broken trajectories by matching detected tracklets with unmatched ones.
    
    This method separates both unmatched and detected tracklets by type (person vs vehicle),
    then performs separate matching for each type using link score matrices and the Hungarian algorithm.
    This ensures person-to-person and vehicle-to-vehicle connections only.
    
    Args:
        current_frame (int): The current frame being processed
        detected_tracklets (list[Track]): New tracklets detected in the current frame
    """

    untracked_vehicle_tracklets_to_match = []
    untracked_person_tracklets_to_match = []

    detected_person_tracklets = []
    detected_vehicle_tracklets = []

    # Filter tracklets by examining if the end frame of their position does is less than the current frame
    for index,tracklet in enumerate(self.untracked_tracklets):
      if current_frame - tracklet.end_frame <= self.lost_track_tresh and current_frame - tracklet.start_frame > 0:

        if tracklet.track_list[-1].get('class_name') == 'person':
          untracked_person_tracklets_to_match.append(index)
        else:
          untracked_vehicle_tracklets_to_match.append(index)

    for index,tracklet in enumerate(detected_tracklets):
      if tracklet.track_list[0].get('class_name') == 'person':
        detected_person_tracklets.append(tracklet)
      else:
        detected_vehicle_tracklets.append(tracklet)



    # Person linking score matching
    self.calculate_link_score_matrix(
        untracked_tracklets_to_match = untracked_person_tracklets_to_match,
        detections_to_match = detected_person_tracklets,
        match_type = 'person'
        )

    # Vehicle linking score matching
    self.calculate_link_score_matrix(
        untracked_tracklets_to_match = untracked_vehicle_tracklets_to_match,
        detections_to_match = detected_vehicle_tracklets,
        match_type = 'vehicle'
      )


  def calculate_link_score_matrix(self,untracked_tracklets_to_match:list[Track],detections_to_match:list[Track],match_type:str):
    """
    Build a link score matrix and use Hungarian algorithm to find optimal tracklet matches.
    
    Process:
    1. Create an NxM cost matrix where N=unmatched tracklets, M=detected tracklets
    2. Fill matrix with (1 - link_score) to convert probabilities to costs (minimize cost)
    3. Apply Hungarian algorithm to find optimal one-to-one assignments
    4. Merge matched tracklets if confidence exceeds positive_match_thresh
    5. Add unmatched detected tracklets as new unmatched tracklets
    
    Args:
        untracked_tracklets_to_match (list[int]): Indices of unmatched tracklets (filtered by type and temporal constraints)
        detections_to_match (list[Track]): Detected tracklets of the same type
        match_type (str): Either 'person' or 'vehicle' to select appropriate ReID model
    """

    if len(untracked_tracklets_to_match) > 0 and len(detections_to_match) > 0:
      x_dim = len(untracked_tracklets_to_match)
      y_dim = len(detections_to_match)

      link_score_matrix = np.zeros((x_dim,y_dim))

      for i in range(x_dim):
        for j in range(y_dim):

          # Compute error rate
          link_score_matrix[i][j] = 1 - self.calculate_link_score(
              self.untracked_tracklets[untracked_tracklets_to_match[i]],
              detections_to_match[j],
              matching_type = match_type
          )


      # Get optimal assignment through hungarian matching
      row_ind, col_ind = linear_sum_assignment(link_score_matrix)

      # Unassignned detections will be added as new untracked tracklets for future frame matching
      for i in range(len(detections_to_match)):
        if i not in col_ind:
          self.untracked_tracklets.append(detections_to_match[i])

      # Update the current untracked_tracklets list
      for i in range(len(row_ind)):
        probability_score = 1 - link_score_matrix[row_ind[i],col_ind[i]]

        if probability_score >= self.positive_match_thresh:
          updated_tracklet = self.combine_trajectories(track1=self.untracked_tracklets[row_ind[i]],track2=detections_to_match[col_ind[i]])
          self.untracked_tracklets[row_ind[i]] = updated_tracklet

        else:
          # If link score threshold does not meet minimum link score, add the detections as untracked tracklets and don't merge
          self.untracked_tracklets.append(detections_to_match[col_ind[i]])

    elif len(untracked_tracklets_to_match) == 0 and len(detections_to_match) > 0:
      self.untracked_tracklets.extend(detections_to_match)


  def calculate_link_score(self,untracked_tracklet:Track,detected_tracklet:Track,matching_type:str):
    """
    Calculate the likelihood that two tracklets represent the same object.
    
    This method extracts 6 features:
    1. IoU: Intersection over Union of bounding boxes
    2. Euclidean Distance: Spatial distance between box centers
    3. Aspect Ratio Width: Width ratio between boxes
    4. Aspect Ratio Height: Height ratio between boxes
    5. Direction: Directional information from bbox displacement
    6. Similarity: Cosine similarity of appearance features (from ReID models)
    
    These features are fed into the pre-trained logistic regression model to predict
    the probability [0-1] that both tracklets belong to the same object.
    
    Args:
        untracked_tracklet (Track): The unmatched tracklet (ending frame)
        detected_tracklet (Track): The newly detected tracklet (starting frame)
        matching_type (str): Either 'person' or 'vehicle'
    
    Returns:
        float: Probability [0-1] that the two tracklets are the same object
    
    Raises:
        Exception: If matching_type is neither 'person' nor 'vehicle'
    """
    if matching_type != 'person' and matching_type != 'vehicle': # Changed 'or' to 'and'
      raise Exception("Matching type not supported. Can only be person or vehicle")

    if matching_type == 'person':
      reid_model = self.person_reid
    else:
      reid_model = self.vehicle_reid

    bbox1 = [
             detected_tracklet.track_list[-1].get('bb_left'),
             detected_tracklet.track_list[-1].get('bb_top'),
             detected_tracklet.track_list[-1].get('bb_width'),
             detected_tracklet.track_list[-1].get('bb_height')
            ]

    bbox2 = [
             detected_tracklet.track_list[0].get('bb_left'),
             detected_tracklet.track_list[0].get('bb_top'),
             detected_tracklet.track_list[0].get('bb_width'),
             detected_tracklet.track_list[0].get('bb_height')
            ]

    appearance_vector1 = extract_appearance_vector_from_frame(self.video_path,untracked_tracklet.track_list[-1].get('frame_number') + 1,reid_model,bbox1)
    appearance_vector2 = extract_appearance_vector_from_frame(self.video_path,detected_tracklet.track_list[0].get('frame_number') + 1, reid_model,bbox2)


    iou = calculate_iou(bbox1,bbox2)
    euclidean_distance = get_euclidean_distance(bbox1,bbox2)
    aspect_ratio_w,aspect_ratio_h = get_bounding_box_ratio(bbox1,bbox2)
    direction = get_direction(bbox1,bbox2)
    similarity = cosine(appearance_vector1.flatten(),appearance_vector2.flatten())

    # Define names in the exact order the model was trained on
    feature_names = ['iou', 'euclidean_distance', 'aspect_ratio_width', 'aspect_ratio_height', 'direction', 'similarity']
    item_data = [[iou, euclidean_distance, aspect_ratio_w, aspect_ratio_h, direction, similarity]]

    # Create a DataFrame for the single row
    df_item = pd.DataFrame(item_data, columns=feature_names)

    # Predict probability score between 0.0 to 1.0 inclusive that they are the same object
    positive_probability_score = self.model.predict_proba(df_item)[0, 1]
    return round(positive_probability_score,4)

  def combine_trajectories(self,track1:Track,track2:Track):
    """
    Merge two tracklets that have been matched together.
    
    The merged tracklet includes:
    - All frames from track1
    - Interpolated frames filling the gap between track1 and track2
    - All frames from track2
    
    Args:
        track1 (Track): The first tracklet (chronologically earlier)
        track2 (Track): The second tracklet (chronologically later)
    
    Returns:
        Track: A new Track object spanning from track1.start_frame to track2.end_frame
    """
    t1 = track1.track_list.copy()
    t2 = track2.track_list.copy()

    interpolated_data = self.interpolate_bbox(track1,track2)

    t1.extend(interpolated_data)
    t1.extend(t2)

    return Track(
        start_frame=track1.start_frame,
        end_frame=track2.end_frame,
        track_list=t1
      )

  def interpolate_bbox(self,track1:Track, track2:Track):
    """
    Generate interpolated bounding box data for frames between two tracklets.
    
    Uses linear interpolation for all bbox coordinates (left, top, width, height).
    For categorical fields (class_name, tracker_id, confidence), a midpoint-based
    assignment is used to smoothly transition from one tracklet's identity to the other.
    
    Args:
        track1 (Track): The first tracklet ending at frame X
        track2 (Track): The second tracklet starting at frame X+gap
    
    Returns:
        list[dict]: List of interpolated detection records for frames between track1 and track2
    """
    interpolated_data = []
    x1 = track1.end_frame
    x2 = track2.start_frame

    y1 = track1.track_list[-1]
    y2 = track2.track_list[0]

    xp = [x1,x2]

    fp_bb_left = [y1.get('bb_left'),y2.get('bb_left')]
    fp_bb_top = [y1.get('bb_top'),y2.get('bb_top')]
    fp_w = [y1.get('bb_width'),y2.get('bb_width')]
    fp_h = [y1.get('bb_height'),y2.get('bb_height')]
    class_list = [y1.get('class_name'),y2.get('class_name')]
    conf_score = [y1.get('confidence'),y2.get('confidence')]
    id = [y1.get('tracker_id'),y2.get('tracker_id')]

    frame_list = np.arange(x1 + 1,x2) # Interpolate for frames between x1 and x2

    # Corrected: Use np.interp for interpolation
    bb_left_list = np.interp(frame_list, xp, fp_bb_left)
    bb_top_list = np.interp(frame_list, xp, fp_bb_top)
    bb_w_list = np.interp(frame_list, xp, fp_w)
    bb_h_list = np.interp(frame_list, xp, fp_h)


    midpoint = len(frame_list) // 2
    categorical_index = 0

    for i in range(len(frame_list)):

      if i == midpoint:
        categorical_index = categorical_index + 1

      interpolated_data.append({
        "frame_number": frame_list[i],
        "tracker_id": id[categorical_index],
        "class_name": class_list[categorical_index],
        "bb_left": bb_left_list[i],
        "bb_top": bb_top_list[i],
        "bb_width": bb_w_list[i],
        "bb_height": bb_h_list[i],
        "x_center": bb_left_list[i] + (bb_w_list[i] / 2),
        "y_center": bb_top_list[i] + (bb_h_list[i] / 2),
        "confidence": conf_score[categorical_index],
        'x':-1,
        'y':-1,
        'z':-1
      })

    return interpolated_data

  def finalize_tracklist(self,tracks:list[Track],csv_filename:str):
    """
    Convert finalized tracks to DataFrame and save to CSV.
    
    For each track, this method:
    1. Performs class rescoring to determine the most confident class across all detections
    2. Assigns a unified tracker ID
    3. Updates the class_name field with the rescored class
    4. Concatenates all tracks into a single DataFrame
    5. Exports the result to the specified CSV file
    
    Args:
        tracks (list[Track]): List of finalized Track objects
        csv_filename (str): Output CSV filename
    """
    tracks.sort(key=lambda x: x.start_frame)
    tracks_df = pd.DataFrame()
    for i,track in enumerate(tracks):
      final_class, _ = self.rescore(track)
      df = pd.DataFrame(track.track_list)
      df['class_name'] = final_class
      df['tracker_id'] = i  + 1

      tracks_df = pd.concat([tracks_df,df])

    tracks_df.to_csv(csv_filename,drop_index=True)

  def rescore(self,track:Track):
    """
    Determine the most confident class for a track by averaging confidence scores.
    
    Iterates through all detections in a track, accumulates confidence scores by class,
    and returns the class with the highest average confidence.
    
    Args:
        track (Track): A Track object with multiple detection records
    
    Returns:
        tuple: (most_confident_class: str, average_confidence: float)
    """
    tracks = track.track_list
    class_conf_dict = dict()
    total_trajectories = len(track.track_list)
    for track in tracks:
      if track.get('class_name') not in class_conf_dict.keys():
        class_conf_dict[track.get('class_name')] = 0

      class_conf_dict[track.get('class_name')] = class_conf_dict[track.get('class_name')] + track.get('confidence')

    for key in class_conf_dict.keys():
      class_conf_dict[key] = class_conf_dict[key] / total_trajectories


    max_class = max(class_conf_dict, key=class_conf_dict.get)
    max_confidence = class_conf_dict[max_class]

    return max_class,round(max_confidence,4)
