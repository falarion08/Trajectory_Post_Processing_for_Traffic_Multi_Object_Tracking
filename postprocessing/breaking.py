import pandas as pd
from tracklet import Track
import numpy as np
class TrajectoryBreakPhase:
  def __init__(self,input_csv_filename:str,video_fps:int,mahalanobis_distance_thresh:float=1.9) -> None:
    self.mot_df = pd.read_csv(input_csv_filename)
    self.video_fps = video_fps
    self.thresh = mahalanobis_distance_thresh

  def process_trajectory(self,filtered_track:pd.DataFrame,distance_list:list[int]):
    filtered_track = filtered_track.reset_index(drop=True) # Ensure contiguous 0-based index and drop old index
    tracklet_list = []
    start_index_copy = 0

    # Iterate through the distances and corresponding track points
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
    An implementation of Kalman Filter using Constant Acceleration Model.
    This predicts the x,y center coordinate of a vehicle trajectory.
  """
  def __init__(self,measurement_period, measurement_error_std, acceleration_std,measured_trajectory_points):
    self.measurement_period = measurement_period
    self.measurement_error_std = measurement_error_std
    self.acceleration_std = acceleration_std
    self.measured_trajectory_points = measured_trajectory_points
    self.starting_frame = measured_trajectory_points[0][0]
    self.ending_frame = measured_trajectory_points[-1][0]
    self.innovation_residual = np.array([])
    self.inv_innovation_covariance = np.array([])



  def initialize_matrices(self):

    dt = self.measurement_period

    # Initialize matrix H
    self.measurement_matrix = np.array([
        [1,0,0,0,0,0],
        [0,0,0,1,0,0]
    ])
    # Create a 6x6 measure for matrix F
    self.state_transition_matrix = np.array([
        [1,dt,0.5*dt**2,0,0,0],
        [0,1,dt,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,dt,0.5*dt**2],
        [0,0,0,0,1,dt],
        [0,0,0,0,0,1]
        ])

    # Initialize matrix Q
    self.process_noise_covariance = np.array([
        [dt**4,dt**3,dt**2,0,0,0],
        [dt**3,dt**2,dt,0,0,0],
        [dt**2,dt,1,0,0,0],
        [0,0,0,dt**4,dt**3,dt**2],
        [0,0,0,dt**3,dt**2,dt],
        [0,0,0,dt**2,dt,1]
    ])
    self.process_noise_covariance = self.process_noise_covariance * self.acceleration_std**2

    # Initialize matrix R
    self.measurement_covariance = np.array([
        [self.measurement_error_std**2,0],
        [0,self.measurement_error_std**2]
    ])

    # Initialize state phase X0,0
    _,first_x,first_y = self.measured_trajectory_points[0]
    # self.state_estimate = np.zeros(6,1)
    self.state_estimate = np.array([first_x,0,0,first_y,0,0])
    self.state_estimate = self.state_estimate.reshape(-1,1)

    # Initalize matrix P0,0
    self.covariance_estimate = np.array([
        [500,0,0,0,0,0],
        [0,500,0,0,0,0],
        [0,0,500,0,0,0],
        [0,0,0,500,0,0],
        [0,0,0,0,500,0],
        [0,0,0,0,0,500]
    ])

  def predict_state(self):
    # Predict the next state
    self.state_estimate = self.state_transition_matrix @ self.state_estimate
    # Predict the next covariance
    self.covariance_estimate = self.state_transition_matrix @ self.covariance_estimate @ self.state_transition_matrix.T + self.process_noise_covariance

  def update_state(self, measurement):

    # Convert measurement tuple to a numpy array (2x1)
    measurement_vector = np.array([[measurement[0]], [measurement[1]]])

    # Compute for the innovation residual and inverse innovation covariance

    self.innovation_residual = measurement_vector - self.measurement_matrix @ self.state_estimate

    self.inv_innovation_covariance = np.linalg.inv(self.measurement_matrix @ self.covariance_estimate @ \
                                self.measurement_matrix.T + self.measurement_covariance)


    # Compute Kalman Gain
    kalman_gain = self.covariance_estimate @ self.measurement_matrix.T @ self.inv_innovation_covariance

    # Update state estimate
    self.state_estimate = self.state_estimate + kalman_gain @ self.innovation_residual

    # Update covariance estimate
    self.covariance_estimate = (np.eye(self.covariance_estimate.shape[0]) - kalman_gain @ self.measurement_matrix) @ self.covariance_estimate

  def run_kalman_filter(self):
    """
    Runs the Kalman filter on the measured trajectory points and compute the mahalanobis distance between
    the actual measurement and the predicted measurement.
    """

    computed_distances = []

    i = 0
    current_frame = self.starting_frame

    self.initialize_matrices()

    # Run the kalman filter for the range of frame that contains the measurements for the unique track

    while current_frame <= self.ending_frame:
      # Prediction step
      self.predict_state()


      frame_number,x_center,y_center = self.measured_trajectory_points[i]

      if current_frame == frame_number:
        # Only update the state when there is a estimate present within the frame
        self.update_state(measurement=(x_center,y_center))

        mahalanobis_distance = np.sqrt(self.innovation_residual.T @ self.inv_innovation_covariance @ self.innovation_residual)
        mahalanobis_distance = mahalanobis_distance[0][0] # Extract the scalar value
        mahalanobis_distance = round(mahalanobis_distance,4) # Round to the nearest 4 decimal places
        computed_distances.append(mahalanobis_distance)

        i = i + 1


      current_frame = current_frame + 1

    return computed_distances




