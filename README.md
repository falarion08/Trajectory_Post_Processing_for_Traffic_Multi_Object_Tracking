# Trajectory Post-Processing for Traffic Multi-Object Tracking

A lightweight toolkit for improving traffic multi-object tracking (MOT) outputs by:
- breaking long trajectories into shorter, reliable tracklets when motion anomalies occur
- linking tracklets back into complete object trajectories using learned pairwise matching
- smoothing and stitching trajectories for cleaner final outputs

## Features
- `TrajectoryBreakPhase`: detects unstable trajectory points with a 2D Kalman filter and Mahalanobis distance
- `KalmanFilter2D`: constant-acceleration model for prediction and anomaly scoring
- `LinkingPhase`: matches tracklets using a logistic regression model and geometric/motion features
- `Track`: lightweight tracklet container for frame intervals and detection records

## Requirements
- Python 3.8+
- `opencv-python`
- `numpy`
- `onnxruntime-gpu`
- `pandas`
- `scikit-learn`
- `scipy`
- `supervision`

> The repository includes a minimal `requirements.txt`, but the code also depends on `pandas`, `scikit-learn`, and `scipy`.

## Installation
Install dependencies with pip:

```bash
pip install -r requirements.txt
pip install pandas scikit-learn scipy supervision
```

## Usage
Example pipeline using the main processing classes:

```python
from trajectory_improvement.postprocessing import TrajectoryBreakPhase, LinkingPhase
from trajectory_improvement.tracklet import Track
from sklearn.linear_model import LogisticRegression
import pickle

# 1. Split long object trajectories into shorter tracklets
break_phase = TrajectoryBreakPhase(input_csv_filename='input_mot.csv', video_fps=30)
tracklets = break_phase.create_trackelts()

# 2. Load a trained logistic regression model for linking
with open('linking_model.pkl', 'rb') as f:
    log_reg_model = pickle.load(f)

# 3. Reconnect broken tracklets and save the final MOT output
link_phase = LinkingPhase(
    log_reg_model=log_reg_model,
    track_list=tracklets,
    csv_filename='output_mot.csv',
    input_csv_filename='input_mot.csv'
)
link_phase.run_post_process()
```

## Expected Input Format
The input MOT CSV should contain at least the following columns:

- `frame_number`
- `tracker_id`
- `x_center`
- `y_center`
- `bb_left`
- `bb_top`
- `bb_width`
- `bb_height`
- `class_name`
- `confidence`

The `LinkingPhase` also relies on class labels and confidence scores to merge tracklets.

## Module Summary
- `trajectory_improvement.postprocessing`
  - `TrajectoryBreakPhase`: processes each object trajectory, computes Mahalanobis distance
    per frame, and splits trajectories at anomalies.
  - `KalmanFilter2D`: performs prediction and update steps for 2D motion and returns
    innovation-based anomaly scores.
  - `LinkingPhase`: uses pairwise features and a logistic regression classifier to link
    tracklets into complete trajectories and exports the final CSV.
- `trajectory_improvement.tracklet`
  - `Track`: container object for a tracklet with start/end frames and detection sequence.

## Notes
- The current implementation uses a constant acceleration motion model and may require tuning
  of the `mahalanobis_distance_thresh`, `lost_track_tresh`, and `positive_match_thresh`
  parameters for optimal results on different datasets.
- The `LinkingPhase` assumes that tracklets with very short lengths should be finalized
  rather than held as linking candidates.
