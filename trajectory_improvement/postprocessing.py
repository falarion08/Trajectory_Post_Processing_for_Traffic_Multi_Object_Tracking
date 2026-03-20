import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from trajectory_improvement.tracklet import Track
from fastreid.reid import FastReID
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

    def __init__(
        self,
        input_csv_filename: str,
        video_fps: int,
        mahalanobis_distance_thresh: float = 1.9,
    ) -> None:
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

    def process_trajectory(
        self, filtered_track: pd.DataFrame, distance_list: list[int]
    ):
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
        filtered_track = filtered_track.reset_index(
            drop=True
        )  # Ensure contiguous 0-based index and drop old index
        tracklet_list = []
        start_index_copy = 0

        # Iterate through the distances and corresponding track points.
        for i in range(len(distance_list)):

            # A break is detected if the current point's Mahalanobis distance is above threshold
            if distance_list[i] >= self.thresh:

                # slice from start_index_copy up to i (exclusive)
                tracklet_segment = filtered_track.iloc[start_index_copy:i]

                if not tracklet_segment.empty:
                    tracklet_records = tracklet_segment.to_dict(orient="records")
                    frame_start = tracklet_records[0]["frame_number"]
                    frame_end = tracklet_records[-1]["frame_number"]

                    tracklet = Track(
                        start_frame=frame_start,
                        end_frame=frame_end,
                        track_list=tracklet_records,
                    )
                    tracklet_list.append(tracklet)

                # The new segment starts from the current point 'i'
                start_index_copy = i

        # After the loop, add the last tracklet segment if it exists
        # This segment runs from the last 'start_index_copy' to the very end of the filtered_track
        tracklet_segment = filtered_track.iloc[start_index_copy : len(filtered_track)]

        if not tracklet_segment.empty:
            tracklet_records = tracklet_segment.to_dict(orient="records")
            frame_start = tracklet_records[0]["frame_number"]
            frame_end = tracklet_records[-1]["frame_number"]

            tracklet = Track(
                start_frame=frame_start,
                end_frame=frame_end,
                track_list=tracklet_records,
            )
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
        track_id_list = self.mot_df["tracker_id"].unique()
        broken_track_list = []

        for track_id in track_id_list:

            filtered_track = (
                self.mot_df[self.mot_df["tracker_id"] == track_id]
                .sort_values(by="frame_number")
                .copy()
            )
            trajectory_list = filtered_track[
                ["frame_number", "x_center", "y_center"]
            ].values.tolist()

            kalman_filter = KalmanFilter2D(
                measurement_period=1 / self.video_fps,
                measurement_error_std=3,
                acceleration_std=0.2,
                measured_trajectory_points=trajectory_list,
            )

            computed_distances = kalman_filter.run_kalman_filter()

            tracklet_list = self.process_trajectory(filtered_track, computed_distances)

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

    This class implements a sophisticated multi-object tracking post-processing pipeline that addresses
    common tracking failures such as occlusions, temporary disappearances, and identity switches.
    It uses a combination of spatial, temporal, and appearance-based features to determine whether
    two tracklets belong to the same object.

    Key Features:
    - **Appearance Matching**: Uses ReID (Re-Identification) models to compare visual features
    - **Spatial Reasoning**: Considers bounding box IoU, Euclidean distance, and aspect ratio similarity
    - **Temporal Constraints**: Enforces maximum time gaps for valid connections
    - **Motion Continuity**: Analyzes direction consistency between tracklets
    - **Machine Learning**: Employs a trained logistic regression model for link probability prediction
    - **Optimal Assignment**: Uses Hungarian algorithm for efficient one-to-one matching

    The linking algorithm processes frames sequentially:
    1. Identifies new tracklets starting in the current frame
    2. Matches them against recently active unmatched tracklets
    3. Computes pairwise similarity scores using multiple features
    4. Applies optimal assignment to maximize overall matching confidence
    5. Merges successfully matched tracklets
    6. Finalizes tracklets that remain unmatched beyond temporal thresholds

    Supports separate handling for different object types (persons vs vehicles) to ensure
    type-consistent matching and prevent cross-category identity transfers.
    """
    def __init__(
        self, 
        video_path: str, 
        log_reg_model: LogisticRegression, 
        track_list: list[Track], 
        csv_filename: str, 
        vehicle_reid_path: str, 
        person_reid_path: str, 
        lost_track_tresh: int = 1,
        positive_match_thresh: float = 0.50,
        candidate_tracklet_characterization_window = 5,
        speed_residual_weight = 0.60
    ) -> None:
        self.candidate_tracklet_characterization_window = candidate_tracklet_characterization_window
        self.speed_residual_weight = speed_residual_weight
        self.model = log_reg_model
        self.frame_dict = dict()
        self.track_list = track_list.copy()
        self.linked_track_list = []
        self.scaler = StandardScaler()
        self.tracklet_linking_candidates = []
        self.video_path = video_path
        video_info = sv.VideoInfo.from_video_path(video_path=self.video_path)
        self.frame_width = video_info.width
        self.frame_height = video_info.height
        self.video_fps = video_info.fps
        self.lost_track_tresh = lost_track_tresh * self.video_fps
        self.positive_match_thresh = positive_match_thresh
        self.csv_filename = csv_filename
        self.person_reid = FastReID(onnx_path=person_reid_path, reid_type="person")
        self.vehicle_reid = FastReID(onnx_path=vehicle_reid_path, reid_type="vehicle")
        if not self.model or not isinstance(self.model, LogisticRegression):
            raise Exception("Model not loaded correctly. Check the model path or model type")
        if len(self.track_list) == 0:
            print("No items to process")
            return
        self.track_list.sort(key=lambda x: x.start_frame, reverse=True)
        self.start_frame = self.track_list[-1].start_frame
        self.end_frame = self.track_list[0].end_frame

    def link_broken_trajectories(self):
        generator = sv.get_video_frames_generator(self.video_path)
        frame_to_map = set()
        for i, frame in enumerate(generator):
            frame_number = i + 1
            if frame_number >= self.start_frame and frame_number <= self.end_frame:
                self.update_tracklet_linking_candidates(frame_number)
                detected_tracklets_from_frame = self.check_new_detections_from_frame(frame_number, frame_to_map)
                if frame_number in frame_to_map:
                    self.frame_dict[frame_number] = frame
                if len(detected_tracklets_from_frame) > 0:
                    if len(self.tracklet_linking_candidates) > 0:
                        self.link_detections_with_candidates(frame_number, detected_tracklets_from_frame)
                    else:
                        for detected_tracklet in detected_tracklets_from_frame:
                            if len(detected_tracklet.track_list) >= 2:
                                self.tracklet_linking_candidates.append(detected_tracklet)
                                frame_to_map.add(detected_tracklet.track_list[-1].get('frame_number'))
        self.linked_track_list.extend(self.tracklet_linking_candidates)
        self.finalize_tracklist(tracks=self.linked_track_list, csv_filename=self.csv_filename)

    def check_new_detections_from_frame(self, frame_ref: int, frame_to_map: set):
        detected_tracklets = []
        while len(self.track_list) > 0 and self.track_list[-1].start_frame == frame_ref:
            tracklet = self.track_list.pop()
            detected_tracklets.append(tracklet)
            frame_to_map.add(tracklet.track_list[0].get('frame_number'))
            frame_to_map.add(tracklet.track_list[-1].get('frame_number'))
        return detected_tracklets

    def update_tracklet_linking_candidates(self, current_frame: int):
        linked_trajectories = []
        updated_tracklet_linking_candidates = []
        if len(self.tracklet_linking_candidates) > 0:
            for tracklet in self.tracklet_linking_candidates:
                if current_frame - tracklet.end_frame > self.lost_track_tresh:
                    linked_trajectories.append(tracklet)
                else:
                    updated_tracklet_linking_candidates.append(tracklet)
        self.linked_track_list.extend(linked_trajectories)
        self.tracklet_linking_candidates = updated_tracklet_linking_candidates

    def link_detections_with_candidates(self, current_frame, detected_tracklets: list[Track]):
        untracked_vehicle_tracklets_to_match = []
        untracked_person_tracklets_to_match = []
        detected_person_tracklets = []
        detected_vehicle_tracklets = []
        for index, tracklet in enumerate(self.tracklet_linking_candidates):
            if current_frame - tracklet.end_frame <= self.lost_track_tresh and current_frame - tracklet.end_frame > 0:
                class_name = tracklet.track_list[-1].get("class_name")
                if class_name == "person":
                    untracked_person_tracklets_to_match.append(index)
                else:
                    untracked_vehicle_tracklets_to_match.append(index)
        for tracklet in detected_tracklets:
            class_name = tracklet.track_list[0].get("class_name")
            if class_name == "person":
                detected_person_tracklets.append(tracklet)
            else:
                detected_vehicle_tracklets.append(tracklet)
        if len(detected_person_tracklets) > 0:
            self.calculate_link_score_matrix(untracked_person_tracklets_to_match, detected_person_tracklets, "person")
        if len(detected_vehicle_tracklets) > 0:
            self.calculate_link_score_matrix(untracked_vehicle_tracklets_to_match, detected_vehicle_tracklets, "vehicle")

    def calculate_link_score_matrix(self, tracklet_linking_candidates_to_match: list[int], detections_to_match: list[Track], match_type: str):
        filtered_candidates = []
        feature_names = ["iou", "euclidean_distance", "aspect_ratio_width", "aspect_ratio_height", "direction_similarity", "similarity"]
        for idx in tracklet_linking_candidates_to_match:
            tracklet = self.tracklet_linking_candidates[idx]
            class_name = tracklet.track_list[-1].get("class_name")
            if match_type == "person" and class_name == "person":
                filtered_candidates.append(idx)
            elif match_type == "vehicle" and class_name != "person":
                filtered_candidates.append(idx)
        tracklet_linking_candidates_to_match = filtered_candidates
        if len(tracklet_linking_candidates_to_match) > 0 and len(detections_to_match) > 0:
            x_dim = len(tracklet_linking_candidates_to_match)
            y_dim = len(detections_to_match)
            link_score_matrix = np.zeros((x_dim, y_dim))
            features_batch = []
            indices = []
            for i in range(x_dim):
                for j in range(y_dim):
                    features = self.extract_pairwise_features(self.tracklet_linking_candidates[tracklet_linking_candidates_to_match[i]], detections_to_match[j], matching_type=match_type)
                    features_batch.append(features)
                    indices.append((i, j))
            if features_batch:
                df_batch = pd.DataFrame(features_batch, columns=feature_names)
                probabilities = self.model.predict_proba(df_batch)[:, 1]
                for idx, (i, j) in enumerate(indices):
                    link_score_matrix[i][j] = 1 - (probabilities[idx] - (self.speed_residual_weight * self.compute_speed_residual(self.tracklet_linking_candidates[tracklet_linking_candidates_to_match[i]], detections_to_match[j])))
            row_ind, col_ind = linear_sum_assignment(link_score_matrix)
            for j in range(len(detections_to_match)):
                if j not in col_ind:
                    if len(detections_to_match[j].track_list) >= 2:
                        self.tracklet_linking_candidates.append(detections_to_match[j])
            for i in range(len(row_ind)):
                probability_score = 1 - link_score_matrix[row_ind[i], col_ind[i]]
                if probability_score >= self.positive_match_thresh:
                    updated_tracklet = self.combine_trajectories(track1=self.tracklet_linking_candidates[tracklet_linking_candidates_to_match[row_ind[i]]], track2=detections_to_match[col_ind[i]])
                    self.tracklet_linking_candidates[tracklet_linking_candidates_to_match[row_ind[i]]] = updated_tracklet
                elif len(detections_to_match[col_ind[i]].track_list) >= 2:
                    self.tracklet_linking_candidates.append(detections_to_match[col_ind[i]])
        else:
            for detection in detections_to_match:
                if len(detection.track_list) >= 2:
                    self.tracklet_linking_candidates.append(detection)

    def extract_pairwise_features(self, untracked_tracklet: Track, detected_tracklet: Track, matching_type: str):
        reid_model = self.person_reid if matching_type == "person" else self.vehicle_reid
        bbox1 = [untracked_tracklet.track_list[-1].get("bb_left"), untracked_tracklet.track_list[-1].get("bb_top"), untracked_tracklet.track_list[-1].get("bb_width"), untracked_tracklet.track_list[-1].get("bb_height")]
        bbox2 = [detected_tracklet.track_list[0].get("bb_left"), detected_tracklet.track_list[0].get("bb_top"), detected_tracklet.track_list[0].get("bb_width"), detected_tracklet.track_list[0].get("bb_height")]
        appearance_vector1 = reid_model.run_inference_on_frame(self.frame_dict[untracked_tracklet.track_list[-1].get("frame_number")])
        appearance_vector2 = reid_model.run_inference_on_frame(self.frame_dict[detected_tracklet.track_list[0].get("frame_number")])
        if len(untracked_tracklet.track_list) >= 2:
            dir1 = get_direction((untracked_tracklet.track_list[-2].get("x_center"), untracked_tracklet.track_list[-2].get("y_center")), (untracked_tracklet.track_list[-1].get("x_center"), untracked_tracklet.track_list[-1].get("y_center")))
        else: dir1 = 0
        dir2 = get_direction((untracked_tracklet.track_list[-1].get("x_center"), untracked_tracklet.track_list[-1].get("y_center")), (detected_tracklet.track_list[0].get("x_center"), detected_tracklet.track_list[0].get("y_center")))
        iou = calculate_iou(bbox1, bbox2)
        euclidean_distance = get_euclidean_distance(bbox1, bbox2)
        aspect_ratio_w, aspect_ratio_h = get_bounding_box_ratio(bbox1, bbox2)
        direction = np.cos(dir2 - dir1)
        similarity = cosine_similarity(appearance_vector1, appearance_vector2)[0][0]
        return [iou, euclidean_distance, aspect_ratio_w, aspect_ratio_h, direction, similarity]

    def compute_speed_residual(self, candidate_tracklet: Track, detected_tracklet: Track) -> float:
        previous_frames = candidate_tracklet.track_list[-1 - self.candidate_tracklet_characterization_window:]
        speed_list = []
        curr_index = 0
        candidate_connection_speed = get_speed(previous_frames[-1], detected_tracklet.track_list[0])
        while curr_index + 1 < len(previous_frames):
            p1 = previous_frames[curr_index]
            p2 = previous_frames[curr_index + 1]
            speed = get_speed(p1, p2)
            speed_list.append(speed)
            curr_index += 1
        if len(speed_list) == 1:
            scaled_result = sigmoid_transform(abs(candidate_connection_speed - speed_list[0]))
        else:
            avg_speed = np.mean(speed_list)
            sample_std = np.std(speed_list, ddof=1) if len(speed_list) > 1 else 1e-6
            scaled_result = sigmoid_transform(abs(candidate_connection_speed - avg_speed) / (sample_std + 1e-6))
        return round(scaled_result - 0.5, 4)

    def combine_trajectories(self, track1: Track, track2: Track):
        t1 = track1.track_list.copy()
        t2 = track2.track_list.copy()
        t1.extend(t2)
        return Track(start_frame=track1.start_frame, end_frame=track2.end_frame, track_list=t1)

    def stitch_tracks(self, track_list: list[dict]):
        final_tracklist = []
        for i in range(1, len(track_list)):
            frame1 = track_list[i-1].get('frame_number')
            frame2 = track_list[i].get('frame_number')
            if frame2 - frame1 > 1:
                final_tracklist.append(track_list[i-1])
                interpolated_data = self.interpolate_from_trajectory(track_list[i-1], track_list[i])
                final_tracklist.extend(interpolated_data)
            else:
                final_tracklist.append(track_list[i-1])
        final_tracklist.append(track_list[-1])
        return final_tracklist

    def interpolate_from_trajectory(self, trajectory_point1: dict, trajectory_point2: dict):
        interpolated_data = []
        x1, x2 = trajectory_point1.get('frame_number'), trajectory_point2.get('frame_number')
        xp = [x1, x2]
        fp_bb_left = [trajectory_point1.get('bb_left'), trajectory_point2.get('bb_left')]
        fp_bb_top = [trajectory_point1.get('bb_top'), trajectory_point2.get('bb_top')]
        fp_w = [trajectory_point1.get('bb_width'), trajectory_point2.get('bb_width')]
        fp_h = [trajectory_point1.get('bb_height'), trajectory_point2.get('bb_height')]
        frame_list = np.arange(x1 + 1, x2)
        bb_left_list = np.interp(frame_list, xp, fp_bb_left)
        bb_top_list = np.interp(frame_list, xp, fp_bb_top)
        bb_w_list = np.interp(frame_list, xp, fp_w)
        bb_h_list = np.interp(frame_list, xp, fp_h)
        midpoint = len(frame_list) // 2
        for i in range(len(frame_list)):
            categorical_index = 0 if i < midpoint else 1
            points = [trajectory_point1, trajectory_point2]
            interpolated_data.append({
                "frame_number": frame_list[i],
                "tracker_id": points[categorical_index].get('tracker_id'),
                "class_name": points[categorical_index].get('class_name'),
                "bb_left": bb_left_list[i], "bb_top": bb_top_list[i], "bb_width": bb_w_list[i], "bb_height": bb_h_list[i],
                "x_center": bb_left_list[i] + (bb_w_list[i] / 2), "y_center": bb_top_list[i] + (bb_h_list[i] / 2),
                "confidence": points[categorical_index].get('confidence'), "x": -1, "y": -1, "z": -1
            })
        return interpolated_data

    def smoothen_trajectory(self, track_df: pd.DataFrame, window_size: int = 15):
        center_cols = ['x_center', 'y_center']
        track_df[center_cols] = track_df[center_cols].rolling(window=window_size, min_periods=1, center=True).mean()
        if 'bb_width' in track_df.columns and 'bb_height' in track_df.columns:
            track_df['bb_left'] = track_df['x_center'] - track_df['bb_width'] / 2
            track_df['bb_top'] = track_df['y_center'] - track_df['bb_height'] / 2
        return track_df

    def finalize_tracklist(self, tracks: list[Track], csv_filename: str):
        tracks.sort(key=lambda x: x.start_frame)
        tracks_df = pd.DataFrame()
        for i, track in enumerate(tracks):
            final_class, _ = self.rescore(track)
            final_tracklist = self.stitch_tracks(track.track_list)
            df = pd.DataFrame(final_tracklist)
            df["class_name"] = final_class
            df["tracker_id"] = i + 1
            df = self.smoothen_trajectory(df.copy())
            tracks_df = pd.concat([tracks_df, df])
        if not tracks_df.empty:
            tracks_df.sort_values(by=["frame_number", "tracker_id"], inplace=True)
            tracks_df.to_csv(csv_filename, index=False)

    def rescore(self, track: Track):
        tracks = track.track_list
        class_conf_dict = dict()
        total_trajectories = len(track.track_list)
        for t in tracks:
            c = t.get("class_name")
            class_conf_dict[c] = class_conf_dict.get(c, 0) + t.get("confidence")
        for key in class_conf_dict.keys():
            class_conf_dict[key] = class_conf_dict[key] / total_trajectories
        max_class = max(class_conf_dict, key=class_conf_dict.get)
        return max_class, round(class_conf_dict[max_class], 4)