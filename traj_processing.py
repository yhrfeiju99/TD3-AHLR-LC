import numpy as np

# Global parameters
LANE_WIDTH = 3.499
CAR_WIDTH = 1.8
TARGET_LANE_THRESHOLD = LANE_WIDTH/2 + CAR_WIDTH/2  
STABLE_DURATION = 2
MIN_SWITCH_TIME = 0.5
MAX_SWITCH_TIME = 15.0

sample_freq = 25
time_per_frame = 1 / sample_freq

WINDOW_SIZE = 3

feature_names = [
    "current_longitudinal_speed",
    "current_lateral_speed",
    "current_lateral_position",
    "gap_with_leading",
    "gap_with_following",
    "gap_with_adjacent_leading",
    "gap_with_adjacent_following",
    "rel_speed_with_leading",
    "rel_speed_with_following",
    "rel_speed_with_adjacent_leading",
    "rel_speed_with_adjacent_following"
]

def get_switch_key_info(trajectory, init_y, init_vy):
    # Extract lateral position and speed
    lateral_pos = trajectory[:, 2]
    lateral_speed = trajectory[:, 1]
    n_frames = len(lateral_pos)
    initial_lateral_pos = lateral_pos[0]
    
    # Identify lane change start frame
    start_mask = (np.abs(lateral_pos - initial_lateral_pos) > init_y) & (lateral_speed > init_vy)
    if not np.any(start_mask):
        return {"valid": False}
    start_frame = np.argmax(start_mask)
    
    # Identify lane change end frame with stable duration check
    end_mask = np.abs(lateral_pos) >= TARGET_LANE_THRESHOLD
    consecutive = np.convolve(end_mask.astype(int), np.ones(STABLE_DURATION), mode='valid') == STABLE_DURATION
    if not np.any(consecutive):
        return {"valid": False}
    end_frame = np.argmax(consecutive) + STABLE_DURATION - 1
    
    # Validate switch time range
    total_switch_time = (end_frame - start_frame) * time_per_frame
    if total_switch_time < MIN_SWITCH_TIME or total_switch_time > MAX_SWITCH_TIME:
        return {"valid": False}
    
    return {
        "valid": True,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "total_switch_time": total_switch_time
    }

lct_lst = []
for i in range(len(data)):
    data0 = data[i]
    data0_info = get_switch_key_info(data0)
    print(data0_info)
    lct_lst.append(data0_info['total_switch_time'])

def trajectory_to_frame_samples(trajectory, key_info):
    # Extract key frame indices
    start_frame = key_info["start_frame"]
    end_frame = key_info["end_frame"]
    n_frames = trajectory.shape[0]
    
    # Initialize sample containers
    frame_X = []
    frame_t = []
    
    # Generate frame-level samples
    for current_frame in range(start_frame, end_frame):
        # Calculate remaining switch time (label)
        remaining_frames = end_frame - current_frame
        remaining_time = remaining_frames * time_per_frame
        frame_t.append(remaining_time)
        
        # Extract sliding window data
        window_start = max(current_frame - WINDOW_SIZE + 1, start_frame)
        window_trajectory = trajectory[window_start:current_frame+1, :]
        
        # Extract statistical features from window
        frame_features = []
        for feat_idx in range(11):
            feat_window = window_trajectory[:, feat_idx]
            frame_features.extend([
                feat_window[-1],
                np.mean(feat_window),
                np.max(feat_window),
                np.min(feat_window),
                np.ptp(feat_window)
            ])
        frame_X.append(frame_features)
    
    return np.array(frame_X), np.array(frame_t)

import numpy as np
import pandas as pd

N_SAMPLES_PER_TRAJECTORY = 50

all_frame_X = []
all_frame_t = []

for traj_idx, traj in enumerate(data):
    key_info = get_switch_key_info(traj)
    if not key_info["valid"]:
        continue
    
    frame_X, frame_t = trajectory_to_frame_samples(traj, key_info)
    n_valid_frames = len(frame_X)
    if n_valid_frames == 0:
        continue
    
    # Balanced sampling for fixed number of samples per trajectory
    if n_valid_frames <= N_SAMPLES_PER_TRAJECTORY:
        repeat_times = N_SAMPLES_PER_TRAJECTORY // n_valid_frames
        remain = N_SAMPLES_PER_TRAJECTORY % n_valid_frames
        sampled_idx = np.tile(np.arange(n_valid_frames), repeat_times)
        if remain > 0:
            sampled_idx = np.concatenate([sampled_idx, np.random.choice(n_valid_frames, remain, replace=False)])
    else:
        sampled_idx = np.linspace(0, n_valid_frames-1, N_SAMPLES_PER_TRAJECTORY, dtype=int)
    
    sampled_X = frame_X[sampled_idx]
    sampled_t = frame_t[sampled_idx]
    
    all_frame_X.extend(sampled_X)
    all_frame_t.extend(sampled_t)

# Convert to numpy arrays
X = np.array(all_frame_X)
t = np.array(all_frame_t)

# Define feature names for interpretability
static_feature_names = []
for feat_name in feature_names:
    static_feature_names.extend([
        f"{feat_name}_current",
        f"{feat_name}_win_mean",
        f"{feat_name}_win_max",
        f"{feat_name}_win_min",
        f"{feat_name}_win_range"
    ])

# Convert to DataFrame and Series
X_df = pd.DataFrame(X, columns=static_feature_names)
t_series = pd.Series(t, name="remaining_switch_time")

# Data cleaning: remove NaN/inf values
X_df = X_df.replace([np.inf, -np.inf], np.nan).dropna()
t_series = t_series[X_df.index]

# Print dataset statistics
print(f"Data processing completed!")
print(f"Valid trajectories: {len([t for t in data if get_switch_key_info(t)['valid']])}")
print(f"Fixed samples per trajectory: {N_SAMPLES_PER_TRAJECTORY}")
print(f"Total final samples: {len(X_df)} (valid trajectories × samples per trajectory)")
print(f"Feature dimension per sample: {X_df.shape[1]} (11 raw features × 5 statistics)")
print(f"Remaining switch time range: {t_series.min():.2f}s ~ {t_series.max():.2f}s")
print(f"Remaining switch time mean: {t_series.mean():.2f}s")

# Optional: Verify sample balance
valid_trajectories = [t for t in data if get_switch_key_info(t)['valid']]
traj_duration = []
traj_contribution = []
for traj in valid_trajectories:
    key_info = get_switch_key_info(traj)
    traj_duration.append(key_info["total_switch_time"])
    traj_contribution.append(N_SAMPLES_PER_TRAJECTORY)