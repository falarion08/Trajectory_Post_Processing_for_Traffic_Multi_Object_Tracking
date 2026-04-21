import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from trajectory_improvement.tracklet import Track
import supervision as sv
from utils.utils import (
    get_speed,
    calculate_iou,
    get_euclidean_distance,
    get_bounding_box_ratio,
    get_direction,    
    sigmoid_transform
)
class TrajectoryBreakPhase:
    """
    TrajectoryBreakPhase is responsible for breaking trajectories into tracklets based on Mahalanobis distance.

    This class uses Kalman filtering to detect anomalies in object trajectories and splits them into smaller tracklets
    when the distance exceeds a threshold, indicating potential occlusions or tracking errors.
    """
    def __init__(self, input_csv_filename: str, video_fps: int, mahalanobis_distance_thresh: float = 5.99) -> None:
        """
        Initialize the TrajectoryBreakPhase with input data and parameters.

        Args:
            input_csv_filename (str): Path to the CSV file containing MOT data.
            video_fps (int): Frames per second of the video.
            mahalanobis_distance_thresh (float): Threshold for Mahalanobis distance to break trajectories.
        """
        self.mot_df = pd.read_csv(input_csv_filename)
        self.video_fps = video_fps
        self.thresh = mahalanobis_distance_thresh

    def process_trajectory(self, filtered_track: pd.DataFrame, distance_list: list[float]):
        """
        Process a filtered track and split it into tracklets based on distance thresholds.

        Args:
            filtered_track (pd.DataFrame): DataFrame containing track data for a single tracker_id.
            distance_list (list[float]): List of Mahalanobis distances corresponding to each point.

        Returns:
            list[Track]: List of Track objects representing the broken tracklets.
        """
        filtered_track = filtered_track.reset_index(drop=True)
        tracklet_list = []
        start_index_copy = 0
        for i in range(len(distance_list)):
            if distance_list[i] >= self.thresh:
                tracklet_segment = filtered_track.iloc[start_index_copy:i]
                if not tracklet_segment.empty:
                    tracklet_records = tracklet_segment.to_dict(orient="records")
                    tracklet = Track(start_frame=tracklet_records[0]["frame_number"], end_frame=tracklet_records[-1]["frame_number"], track_list=tracklet_records)
                    tracklet_list.append(tracklet)
                start_index_copy = i
        tracklet_segment = filtered_track.iloc[start_index_copy:]
        if not tracklet_segment.empty:
            tracklet_records = tracklet_segment.to_dict(orient="records")
            tracklet = Track(start_frame=tracklet_records[0]["frame_number"], end_frame=tracklet_records[-1]["frame_number"], track_list=tracklet_records)
            tracklet_list.append(tracklet)
        return tracklet_list

    def create_trackelts(self):
        """
        Create tracklets by processing all tracks in the MOT data.

        Applies Kalman filtering to each unique tracker_id and breaks trajectories into tracklets.

        Returns:
            list[Track]: List of all broken tracklets from all tracks.
        """
        track_id_list = self.mot_df["tracker_id"].unique()
        broken_track_list = []
        for track_id in track_id_list:
            filtered_track = self.mot_df[self.mot_df["tracker_id"] == track_id].sort_values(by="frame_number").copy()
            trajectory_list = filtered_track[["frame_number", "x_center", "y_center"]].values.tolist()
            kalman_filter = KalmanFilter2D(measurement_period=1/self.video_fps, measurement_error_std=10, acceleration_std=1.5, measured_trajectory_points=trajectory_list)
            computed_distances, _ = kalman_filter.run_kalman_filter()
            tracklet_list = self.process_trajectory(filtered_track, computed_distances)
            if tracklet_list: broken_track_list.extend(tracklet_list)
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

    def __init__(
        self,
        measurement_period,
        measurement_error_std,
        acceleration_std,
        measured_trajectory_points,
    ):
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
        self.measurement_matrix = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        # Create 6x6 State Transition Matrix F for constant acceleration model
        # Assumes constant acceleration; position is updated based on velocity and acceleration
        self.state_transition_matrix = np.array(
            [
                [1, dt, 0.5 * dt**2, 0, 0, 0],
                [0, 1, dt, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, dt, 0.5 * dt**2],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # Initialize matrix Q (process noise covariance)
        # Scales with acceleration_std; higher values allow more freedom in motion
        self.process_noise_covariance = np.array(
            [
                [dt**4, dt**3, dt**2, 0, 0, 0],
                [dt**3, dt**2, dt, 0, 0, 0],
                [dt**2, dt, 1, 0, 0, 0],
                [0, 0, 0, dt**4, dt**3, dt**2],
                [0, 0, 0, dt**3, dt**2, dt],
                [0, 0, 0, dt**2, dt, 1],
            ]
        )
        self.process_noise_covariance = (
            self.process_noise_covariance * self.acceleration_std**2
        )

        # Initialize matrix R (measurement covariance)
        # Higher values mean less trust in measurements
        self.measurement_covariance = np.array(
            [[self.measurement_error_std**2, 0], [0, self.measurement_error_std**2]]
        )

        # Initialize state estimate X (6D vector: [x, vx, ax, y, vy, ay])
        # Initialize with first measurement position, zero velocity and acceleration
        _, first_x, first_y = self.measured_trajectory_points[0]
        self.state_estimate = np.array([first_x, 0, 0, first_y, 0, 0])
        self.state_estimate = self.state_estimate.reshape(-1, 1)

        # Initialize covariance estimate P (6x6 matrix representing state uncertainty)
        # High initial values (500) indicate we start with high uncertainty that decreases with measurements
        self.covariance_estimate = np.array(
            [
                [500, 0, 0, 0, 0, 0],
                [0, 500, 0, 0, 0, 0],
                [0, 0, 500, 0, 0, 0],
                [0, 0, 0, 500, 0, 0],
                [0, 0, 0, 0, 500, 0],
                [0, 0, 0, 0, 0, 500],
            ]
        )

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
        self.covariance_estimate = (
            self.state_transition_matrix
            @ self.covariance_estimate
            @ self.state_transition_matrix.T
            + self.process_noise_covariance
        )

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
        self.innovation_residual = (
            measurement_vector - self.measurement_matrix @ self.state_estimate
        )

        # Compute inverse of innovation covariance matrix
        # S = H*P*H^T + R (total uncertainty in measurement space)
        self.inv_innovation_covariance = np.linalg.inv(
            self.measurement_matrix
            @ self.covariance_estimate
            @ self.measurement_matrix.T
            + self.measurement_covariance
        )

        # Compute Kalman Gain: how much to trust the measurement vs prediction
        kalman_gain = (
            self.covariance_estimate
            @ self.measurement_matrix.T
            @ self.inv_innovation_covariance
        )

        # Update state estimate with weighted innovation
        self.state_estimate = (
            self.state_estimate + kalman_gain @ self.innovation_residual
        )

        # Update covariance estimate: reduce uncertainty after incorporating measurement
        self.covariance_estimate = (
            np.eye(self.covariance_estimate.shape[0])
            - kalman_gain @ self.measurement_matrix
        ) @ self.covariance_estimate

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
            frame_number, x_center, y_center = self.measured_trajectory_points[i]

            if current_frame == frame_number:
                # Update step: incorporate measurement into state estimate
                self.update_state(measurement=(x_center, y_center))

                # Compute Mahalanobis distance: how many standard deviations away is the measurement?
                mahalanobis_distance = np.sqrt(
                    self.innovation_residual.T
                    @ self.inv_innovation_covariance
                    @ self.innovation_residual
                )
                mahalanobis_distance = mahalanobis_distance[0][
                    0
                ]  # Extract the scalar value
                mahalanobis_distance = round(
                    mahalanobis_distance, 4
                )  # Round to 4 decimal places
                computed_distances.append(mahalanobis_distance)

                i = i + 1

            # Move to next frame
            current_frame = current_frame + 1

        return computed_distances
class LinkingPhase:
    """
    LinkingPhase is responsible for reconnecting broken trajectory fragments (tracklets) into complete object trajectories.
    """
    def __init__(
        self,
        log_reg_model: LogisticRegression,
        track_list: list[Track],
        csv_filename: str,
        input_csv_filename: str,
        lost_track_tresh: int = 30,
        positive_match_thresh: float = 0.50,
        candidate_tracklet_characterization_window = 5,
        speed_residual_weight = 0.60
    ) -> None:

        self.candidate_tracklet_characterization_window = candidate_tracklet_characterization_window
        self.speed_residual_weight = speed_residual_weight
        self.model = log_reg_model

        self.original_track_list = track_list.copy() # Keep a master copy
        self.track_list = []
        self.linked_track_list = []
        self.tracklet_linking_candidates = []
        self.lost_track_tresh = lost_track_tresh
        self.positive_match_thresh = positive_match_thresh
        self.csv_filename = csv_filename
        self.input_csv_filename = input_csv_filename

        if not self.model or not isinstance(self.model, LogisticRegression):
            raise Exception("Model not loaded correctly. Check the model path or model type")
        if len(self.original_track_list) == 0:
            print("No items to process")
            return

        self.original_track_list.sort(key=lambda x: x.start_frame, reverse=True)
        self.start_frame = self.original_track_list[-1].start_frame
        self.end_frame = self.original_track_list[0].end_frame

    def run_post_process(self):
        """
        Run the post-processing pipeline for linking tracklets into complete trajectories.

        Processes tracks by class, links broken trajectories, and finalizes the output.
        """
        input_df = pd.read_csv(self.input_csv_filename)
        unique_classes = input_df['class_name'].unique().tolist()

        all_final_tracks = []

        for object_class in unique_classes:

            # Filter tracks belonging to the current class using rescored class name
            self.track_list = [t for t in self.original_track_list if self.rescore(t)[0] == object_class]

            if not self.track_list:
                continue

            self.linked_track_list = []
            self.tracklet_linking_candidates = []

            # Re-sort for the processing logic
            self.track_list.sort(key=lambda x: x.start_frame, reverse=True)

            self.link_broken_trajectories()
            all_final_tracks.extend(self.linked_track_list)

        self.finalize_tracklist(tracks=all_final_tracks, csv_filename=self.csv_filename)

    def link_broken_trajectories(self):
        """
        Link broken trajectories by processing frames sequentially.

        Iterates through frames, updates candidates, checks for new detections, and links them.
        """
        frame_number = self.start_frame
        while frame_number <= self.end_frame and len(self.track_list) > 0:
            self.update_tracklet_linking_candidates(frame_number)
            detected_tracklets_from_frame = self.check_new_detections_from_frame(frame_number)

            if len(detected_tracklets_from_frame) > 0:
                if len(self.tracklet_linking_candidates) > 0:
                    self.link_detections_with_candidates(frame_number, detected_tracklets_from_frame)
                else:
                    for detected_tracklet in detected_tracklets_from_frame:
                        if len(detected_tracklet.track_list) >= 2:
                            self.tracklet_linking_candidates.append(detected_tracklet)
                        else:
                            self.linked_track_list.append(detected_tracklet)
            frame_number += 1

        self.linked_track_list.extend(self.tracklet_linking_candidates)

    def check_new_detections_from_frame(self, frame_ref: int):
        """
        Check for new tracklet detections starting at the given frame.

        Args:
            frame_ref (int): The frame number to check.

        Returns:
            list[Track]: List of tracklets starting at this frame.
        """
        detected_tracklets = []
        while len(self.track_list) > 0 and self.track_list[-1].start_frame == frame_ref:
            tracklet = self.track_list.pop()
            detected_tracklets.append(tracklet)
        return detected_tracklets

    def update_tracklet_linking_candidates(self, current_frame: int):
        """
        Update the list of tracklet linking candidates by removing lost tracks.

        Args:
            current_frame (int): The current frame number.
        """
        linked_trajectories = []
        updated_tracklet_linking_candidates = []
        for tracklet in self.tracklet_linking_candidates:
            if current_frame - tracklet.end_frame > self.lost_track_tresh:
                linked_trajectories.append(tracklet)
            else:
                updated_tracklet_linking_candidates.append(tracklet)
        self.linked_track_list.extend(linked_trajectories)
        self.tracklet_linking_candidates = updated_tracklet_linking_candidates

    def link_detections_with_candidates(self, current_frame, detected_tracklets: list[Track]):
        """
        Link detected tracklets with existing candidates.

        Args:
            current_frame: The current frame number.
            detected_tracklets (list[Track]): List of newly detected tracklets.
        """
        # Since run_post_process already filters by class, we link all candidates in this batch
        indices_to_match = list(range(len(self.tracklet_linking_candidates)))
        if indices_to_match:
            self.calculate_link_score_matrix(indices_to_match, detected_tracklets)

    def calculate_link_score_matrix(self, tracklet_linking_candidates_to_match: list[int], detections_to_match: list[Track]):
        """
        Calculate the link score matrix and perform matching using the Hungarian algorithm.

        Args:
            tracklet_linking_candidates_to_match (list[int]): Indices of candidates to match.
            detections_to_match (list[Track]): List of detected tracklets to match.
        """
        feature_names = ["iou", "euclidean_distance", "aspect_ratio_width", "aspect_ratio_height", "direction_similarity"]

        x_dim = len(tracklet_linking_candidates_to_match)
        y_dim = len(detections_to_match)
        link_score_matrix = np.zeros((x_dim, y_dim))
        features_batch = []
        indices = []

        for i in range(x_dim):
            for j in range(y_dim):
                features = self.extract_pairwise_features(self.tracklet_linking_candidates[tracklet_linking_candidates_to_match[i]], detections_to_match[j])
                features_batch.append(features)
                indices.append((i, j))

        if features_batch:
            df_batch = pd.DataFrame(features_batch, columns=feature_names)
            probabilities = self.model.predict_proba(df_batch)[:, 1]

            for idx, (i, j) in enumerate(indices):
                speed_res = self.compute_speed_residual(self.tracklet_linking_candidates[tracklet_linking_candidates_to_match[i]], detections_to_match[j])
                link_score_matrix[i][j] = 1 - (probabilities[idx] - (self.speed_residual_weight * speed_res))

        row_ind, col_ind = linear_sum_assignment(link_score_matrix)

        matched_detections = set()
        for i, j in zip(row_ind, col_ind):
            probability_score = 1 - link_score_matrix[i, j]
            if probability_score >= self.positive_match_thresh:
                updated_tracklet = self.combine_trajectories(self.tracklet_linking_candidates[tracklet_linking_candidates_to_match[i]], detections_to_match[j])
                self.tracklet_linking_candidates[tracklet_linking_candidates_to_match[i]] = updated_tracklet
                matched_detections.add(j)

        for j, detection in enumerate(detections_to_match):
            if j not in matched_detections:
                if len(detection.track_list) >= 2:
                    self.tracklet_linking_candidates.append(detection)
                else:
                    self.linked_track_list.append(detection)

    def extract_pairwise_features(self, untracked_tracklet: Track, detected_tracklet: Track):
        """
        Extract pairwise features between two tracklets for linking.

        Args:
            untracked_tracklet (Track): The candidate tracklet.
            detected_tracklet (Track): The detected tracklet.

        Returns:
            list: List of features [iou, euclidean_distance, aspect_ratio_width, aspect_ratio_height, direction_similarity].
        """
        bbox1 = [untracked_tracklet.track_list[-1].get("bb_left"), untracked_tracklet.track_list[-1].get("bb_top"), untracked_tracklet.track_list[-1].get("bb_width"), untracked_tracklet.track_list[-1].get("bb_height")]
        bbox2 = [detected_tracklet.track_list[0].get("bb_left"), detected_tracklet.track_list[0].get("bb_top"), detected_tracklet.track_list[0].get("bb_width"), detected_tracklet.track_list[0].get("bb_height")]

        if len(untracked_tracklet.track_list) >= 2:
            dir1 = get_direction((untracked_tracklet.track_list[-2].get("x_center"), untracked_tracklet.track_list[-2].get("y_center")), (untracked_tracklet.track_list[-1].get("x_center"), untracked_tracklet.track_list[-1].get("y_center")))
        else: dir1 = 0
        dir2 = get_direction((untracked_tracklet.track_list[-1].get("x_center"), untracked_tracklet.track_list[-1].get("y_center")), (detected_tracklet.track_list[0].get("x_center"), detected_tracklet.track_list[0].get("y_center")))

        return [calculate_iou(bbox1, bbox2), get_euclidean_distance(bbox1, bbox2), *get_bounding_box_ratio(bbox1, bbox2), np.cos(dir2 - dir1)]

    def compute_speed_residual(self, candidate_tracklet: Track, detected_tracklet: Track) -> float:
        """
        Compute the speed residual between candidate and detected tracklets.

        Args:
            candidate_tracklet (Track): The candidate tracklet.
            detected_tracklet (Track): The detected tracklet.

        Returns:
            float: The speed residual value.
        """
        previous_frames = candidate_tracklet.track_list[-1 - self.candidate_tracklet_characterization_window:]
        # Filter out same-frame transitions in history to avoid division by zero
        speed_list = [get_speed(previous_frames[i], previous_frames[i+1]) for i in range(len(previous_frames)-1) if previous_frames[i+1].get('frame_number') != previous_frames[i].get('frame_number')]

        # Check if detection starts at the exact same frame as the candidate end
        if detected_tracklet.track_list[0].get('frame_number') == previous_frames[-1].get('frame_number'):
            return 1.0 # Return maximum residual penalty for temporal overlap

        candidate_connection_speed = get_speed(previous_frames[-1], detected_tracklet.track_list[0])

        if not speed_list:
            return 0.0

        avg_speed = np.mean(speed_list)
        sample_std = np.std(speed_list, ddof=1) if len(speed_list) > 1 else 1e-6
        scaled_result = sigmoid_transform(abs(candidate_connection_speed - avg_speed) / (sample_std + 1e-6))
        return round(scaled_result - 0.5, 4)

    def combine_trajectories(self, track1: Track, track2: Track):
        """
        Combine two tracklets into a single track.

        Args:
            track1 (Track): The first tracklet.
            track2 (Track): The second tracklet to be appended.

        Returns:
            Track: The combined track.
        """
        t1 = track1.track_list.copy()
        t1.extend(track2.track_list)
        return Track(start_frame=track1.start_frame, end_frame=track2.end_frame, track_list=t1)

    def stitch_tracks(self, track_list: list[dict]):
        """
        Stitch track points by interpolating missing frames.

        Args:
            track_list (list[dict]): List of track points.

        Returns:
            list: The stitched track list with interpolated points.
        """
        final_tracklist = []
        for i in range(1, len(track_list)):
            final_tracklist.append(track_list[i-1])
            if track_list[i].get('frame_number') - track_list[i-1].get('frame_number') > 1:
                final_tracklist.extend(self.interpolate_from_trajectory(track_list[i-1], track_list[i]))
        final_tracklist.append(track_list[-1])
        return final_tracklist

    def interpolate_from_trajectory(self, p1: dict, p2: dict):
        """
        Interpolate track points between two frames.

        Args:
            p1 (dict): The first track point.
            p2 (dict): The second track point.

        Returns:
            list: List of interpolated track points.
        """
        interpolated = []
        x1, x2 = p1.get('frame_number'), p2.get('frame_number')
        xp, frame_list = [x1, x2], np.arange(x1 + 1, x2)

        keys = ['bb_left', 'bb_top', 'bb_width', 'bb_height']
        interp_vals = {k: np.interp(frame_list, xp, [p1.get(k), p2.get(k)]) for k in keys}

        midpoint = len(frame_list) // 2
        for i, f in enumerate(frame_list):
            ref = p1 if i < midpoint else p2
            interpolated.append({
                "frame_number": f, "tracker_id": ref.get('tracker_id'), "class_name": ref.get('class_name'),
                **{k: interp_vals[k][i] for k in keys},
                "x_center": interp_vals['bb_left'][i] + interp_vals['bb_width'][i]/2,
                "y_center": interp_vals['bb_top'][i] + interp_vals['bb_height'][i]/2,
                "confidence": ref.get('confidence'), "x": -1, "y": -1, "z": -1
            })
        return interpolated

    def smoothen_trajectory(self, track_df: pd.DataFrame, window_size: int = 15):
        """
        Smooth the trajectory using rolling mean.

        Args:
            track_df (pd.DataFrame): The track DataFrame.
            window_size (int): The window size for smoothing.

        Returns:
            pd.DataFrame: The smoothed track DataFrame.
        """
        center_cols = ['x_center', 'y_center']
        track_df[center_cols] = track_df[center_cols].rolling(window=window_size, min_periods=1, center=True).mean()
        track_df['bb_left'] = track_df['x_center'] - track_df['bb_width'] / 2
        track_df['bb_top'] = track_df['y_center'] - track_df['bb_height'] / 2
        return track_df

    def finalize_tracklist(self, tracks: list[Track], csv_filename: str):
        """
        Finalize the track list by stitching, smoothing, and saving to CSV.

        Args:
            tracks (list[Track]): List of tracks to finalize.
            csv_filename (str): Output CSV filename.
        """
        tracks.sort(key=lambda x: x.start_frame)
        dfs = []
        for i, track in enumerate(tracks):
            final_class, _ = self.rescore(track)
            df = pd.DataFrame(self.stitch_tracks(track.track_list))
            df["class_name"], df["tracker_id"] = final_class, i + 1
            dfs.append(self.smoothen_trajectory(df))

        if dfs:
            final_df = pd.concat(dfs).sort_values(by=["frame_number", "tracker_id"])
            final_df.to_csv(csv_filename, index=False)
            print(f'Saved to: {csv_filename}')

    def rescore(self, track: Track):
        """
        Rescore the class for a track based on confidence.

        Args:
            track (Track): The track to rescore.

        Returns:
            tuple: (class_name, average_confidence)
        """
        class_conf = {}
        for t in track.track_list:
            c = t.get("class_name")
            class_conf[c] = class_conf.get(c, 0) + t.get("confidence")

        max_class = max(class_conf, key=lambda k: class_conf[k]/len(track.track_list))
        return max_class, round(class_conf[max_class]/len(track.track_list), 4)