import numpy as np
import lightgbm as lgb
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class EnvConfig:
    """Environment configuration container"""
    TTC_threshold: float = 3.001
    MaxCrsLoc: float = 3.2
    MaxChangeLaneTime: int = 225
    timeWindow: int = 3
    decel_penalty_weight: float = 10.0
    mu_model_path: str = ".save/lct_mean_pred.txt"
    sigma_model_path: str = ".save/lct_sig_pred.txt"
    delta_t: float = 0.04
    action_Bound: int = 2
    Cpenalty: int = 1000
    Tpenalty: int = 1000
    tau: float = 0.8
    min_remaining_frames: int = 5

@dataclass
class IDMParams:
    """IDM model parameters"""
    a_max: float = 1.5
    b: float = 2.0
    v0: float = 28.0
    s0: float = 2.0
    T: float = 1.2
    delta: float = 4.0

def trajectory_to_frame_samples_fixed_window(trajectory: np.ndarray) -> np.ndarray:
    if not isinstance(trajectory, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(trajectory)}")
    if trajectory.shape != (3, 11):
        raise ValueError(f"Expected shape (3,11), got {trajectory.shape}")
    
    frame_features = []
    for feat_idx in range(11):
        feat_window = trajectory[:, feat_idx]
        frame_features.extend([
            feat_window[-1], np.mean(feat_window), np.max(feat_window),
            np.min(feat_window), np.ptp(feat_window)
        ])
    return np.array(frame_features).reshape(1, -1)

def predict_lct(model_mu: lgb.Booster, model_sigma: lgb.Booster, new_data: np.ndarray) -> Tuple[float, float]:
    pred_remaining_sec = model_mu.predict(new_data)[0]
    pred_sigma2_sec = model_sigma.predict(new_data)[0]
    pred_sigma2_sec = np.maximum(pred_sigma2_sec, 1e-6)
    pred_sigma_sec = np.sqrt(pred_sigma2_sec)
    return pred_remaining_sec, pred_sigma_sec

def idm_acceleration(v: float, v_lead: float, s: float, params: IDMParams) -> float:
    s_des = params.s0 + v*params.T + (v*(v - v_lead))/(2*np.sqrt(params.a_max*params.b))
    s_des = max(s_des, params.s0)
    acc = params.a_max * (1 - (v/params.v0)**params.delta - (s_des/s)**2)
    return acc

class Env:
    def __init__(self, config: Optional[EnvConfig] = None, idm_params: Optional[IDMParams] = None):
        self.config = config or EnvConfig()
        self.idm_params = idm_params or IDMParams()

        self.n_actions = 2
        self.n_features = self.config.timeWindow * 11
 
        try:
            self.model_mu = lgb.Booster(model_file=self.config.mu_model_path)
            self.model_sigma = lgb.Booster(model_file=self.config.sigma_model_path)
            print(f"Models loaded successfully:\n  - Mean: {self.config.mu_model_path}\n  - Variance: {self.config.sigma_model_path}")
        except Exception as e:
            raise FileNotFoundError(f"Model loading failed: {str(e)}") from e
 
        self._initialize_state_containers()
        self._initialize_lane_change_prediction_state()
        
    def _initialize_state_containers(self) -> None:
        self.OLVSpeed: np.ndarray = np.array([])
        self.TLVSpeed: np.ndarray = np.array([])
        
        # Simulation state arrays
        self.LonSimSpeed: np.ndarray = np.array([])
        self.CrsSimSpeed: np.ndarray = np.array([])
        self.CRSSimSpace: np.ndarray = np.array([])
        self.OLVSimSpace: np.ndarray = np.array([])
        self.OFVSimSpace: np.ndarray = np.array([])
        self.TLVSimSpace: np.ndarray = np.array([])
        self.TFVSimSpace: np.ndarray = np.array([])
        
        # Rear vehicle state
        self.OFV_current_speed: Optional[float] = None
        self.TFV_current_speed: Optional[float] = None
        self.OFV_prev_speed: Optional[float] = None
        self.TFV_prev_speed: Optional[float] = None
        
        # Space metrics
        self.OFV_OLV_space: Optional[float] = None
        self.TFV_TLV_space: Optional[float] = None
        
        # Runtime flags
        self.timeStep: int = 0
        self.haveChangedLane: int = 0
        self.isCollision: int = 0
        self.isStall: int = 0
        self.TimeLen: int = 0
        self.lastAction: Tuple[float, float] = (0.0, 0.0)
        self.s: np.ndarray = np.array([])
        self.currentState: List[float] = []
        
        # TTC metrics
        self.OL_TTC: float = 0.0
        self.OF_TTC: float = 0.0
        self.TL_TTC: float = 4.0
        self.TF_TTC: float = 4.0

    def _initialize_lane_change_prediction_state(self) -> None:
        self.lane_change_start_frame: int = 0
        self.history_mu: Optional[float] = None
        self.history_sigma: Optional[float] = None
        self.fixed_completion_time: Optional[float] = None
        self.fixed_sigma: Optional[float] = None

    def reset(self, data: np.ndarray) -> np.ndarray:
        # Validate input data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array for data, got {type(data)}")
        
        # Reset core state
        self._initialize_state_containers()
        self._initialize_lane_change_prediction_state()
        
        self.timeStep = self.config.timeWindow
        self.TimeLen = data.shape[0]
        
        # Load front vehicle speed data
        self.OLVSpeed = data[:, 11] if self.TimeLen > 0 else np.array([self.idm_params.v0])
        self.TLVSpeed = data[:, 13] if self.TimeLen > 0 else np.array([self.idm_params.v0])
        
        # Initialize simulation arrays
        min_len = max(self.TimeLen, self.timeStep)
        self.LonSimSpeed = np.zeros(min_len, dtype=np.float64)
        self.CrsSimSpeed = np.zeros(min_len, dtype=np.float64)
        self.CRSSimSpace = np.zeros(min_len, dtype=np.float64)
        self.OLVSimSpace = np.zeros(min_len, dtype=np.float64)
        self.OFVSimSpace = np.zeros(min_len, dtype=np.float64)
        self.TLVSimSpace = np.zeros(min_len, dtype=np.float64)
        self.TFVSimSpace = np.zeros(min_len, dtype=np.float64)
 
        # if self.TimeLen > 0:
        #     init_vals = [
        #         data[0, 0], data[0, 1], data[0, 2],
        #         data[0, 3], data[0, 4], data[0, 5],
        #         data[0, 6]
        #     ]
        # else:
        #     init_vals = [20.0, 0.0, 0.0, 50.0, -30.0, 60.0, -40.0]
        
        self.LonSimSpeed[0] = init_vals[0]
        self.CrsSimSpeed[0] = init_vals[1]
        self.CRSSimSpace[0] = init_vals[2]
        self.OLVSimSpace[0] = init_vals[3]
        self.OFVSimSpace[0] = init_vals[4]
        self.TLVSimSpace[0] = init_vals[5]
        self.TFVSimSpace[0] = init_vals[6]
 
        self.OFV_current_speed = data[0, 12] if (self.TimeLen > 0 and data.shape[1] > 12) else self.idm_params.v0
        self.TFV_current_speed = data[0, 14] if (self.TimeLen > 0 and data.shape[1] > 14) else self.idm_params.v0
        self.OFV_prev_speed = self.OFV_current_speed
        self.TFV_prev_speed = self.TFV_current_speed
 
        self.OFV_OLV_space = max(self.OFVSimSpace[0] - self.OLVSimSpace[0], self.idm_params.s0)
        self.TFV_TLV_space = max(self.TLVSimSpace[0] - self.TFVSimSpace[0], self.idm_params.s0)

        if self.TimeLen >= self.config.timeWindow:
            temp = data[:self.config.timeWindow, :11]
        else:
            temp = np.tile(data[0:1, :11], (self.config.timeWindow, 1))
        
        self.s = temp.reshape(-1).squeeze()
        self.currentState = self.s[-11:].tolist()
 
        OLVReSpd = self.currentState[7]
        OFVReSpd = self.currentState[8]
        self.OL_TTC = -self.currentState[3] / (OLVReSpd + 0.05)
        self.OF_TTC = -self.currentState[4] / (OFVReSpd + 0.05)
        
        return self.s

    def step(self, action: Tuple[float, float]) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        self.timeStep += 1
        
        ego_prev_speed = self.currentState[0]
        ego_prev_OLV_space = self.currentState[3]
        ego_prev_TLV_space = self.currentState[5]
        ego_prev_OFV_space = self.currentState[4]
        ego_prev_TFV_space = self.currentState[6]
        
        LonSpd = self.currentState[0] + action[0] * self.config.delta_t
        CrsSpd = self.currentState[1] + action[1] * self.config.delta_t
        LonSpd = max(LonSpd, 1e-5)
        self.isStall = 1 if LonSpd <= 1e-4 else 0
        CrsLoc = self.currentState[2] + CrsSpd * self.config.delta_t
 
        onOLane = CrsLoc <= 2.5
        onTLane = CrsLoc > 3.2
        on_lane_changing = not (2.5 < CrsLoc < 3.2)
        
        # LCT reward
        if (self.lane_change_start_frame is not None and 
            CrsLoc < self.config.MaxCrsLoc and 
            self.fixed_completion_time is None):

            if len(self.s) != self.config.timeWindow * 11:
                self.s = np.tile(self.currentState, self.config.timeWindow)[:self.config.timeWindow*11]

            trajectory = self.s.reshape([3, 11])
            win_features = trajectory_to_frame_samples_fixed_window(trajectory)
            pred_remaining_sec, pred_sigma_sec = predict_lct(self.model_mu, self.model_sigma, win_features)
            
            pred_remaining_frames = pred_remaining_sec / self.config.delta_t
            pred_sigma_frames = pred_sigma_sec / self.config.delta_t
            
            used_frames = self.timeStep - self.lane_change_start_frame
            current_pred_total_frames = used_frames + pred_remaining_frames
            
            if pred_remaining_frames < self.config.min_remaining_frames:
                self.fixed_completion_time = current_pred_total_frames
                self.fixed_sigma = pred_sigma_frames
            else:
                if self.history_mu is None or self.history_sigma is None:
                    self.history_mu = current_pred_total_frames
                    self.history_sigma = pred_sigma_frames
                else:
                    self.history_mu = self.config.tau * current_pred_total_frames + (1 - self.config.tau) * self.history_mu
                    self.history_sigma = self.config.tau * pred_sigma_frames + (1 - self.config.tau) * self.history_sigma
 
        OLVSpd_idx = min(self.timeStep - 1, len(self.OLVSpeed) - 1)
        TLVSpd_idx = min(self.timeStep - 1, len(self.TLVSpeed) - 1)
        OLVSpd = self.OLVSpeed[OLVSpd_idx]
        TLVSpd = self.TLVSpeed[TLVSpd_idx]
 
        if onOLane or (on_lane_changing and CrsLoc <= 2.8):
            ofv_lead_speed = ego_prev_speed
            ofv_lead_space = ego_prev_OFV_space
        else:
            ofv_lead_speed = OLVSpd
            ofv_lead_space = self.OFV_OLV_space
        
        if onTLane or (on_lane_changing and CrsLoc >= 2.8):
            tfv_lead_speed = ego_prev_speed
            tfv_lead_space = ego_prev_TFV_space
        else:
            tfv_lead_speed = TLVSpd
            tfv_lead_space = self.TFV_TLV_space
        
        ofv_lead_space = max(ofv_lead_space, self.idm_params.s0)
        tfv_lead_space = max(tfv_lead_space, self.idm_params.s0)
 
        ofv_prev_speed = self.OFV_current_speed
        a_ofv = idm_acceleration(ofv_prev_speed, ofv_lead_speed, ofv_lead_space, self.idm_params)
        OFVSpd = ofv_prev_speed + a_ofv * self.config.delta_t
        OFVSpd = max(OFVSpd, 0.0)
        
        tfv_prev_speed = self.TFV_current_speed
        a_tfv = idm_acceleration(tfv_prev_speed, tfv_lead_speed, tfv_lead_space, self.idm_params)
        TFVSpd = tfv_prev_speed + a_tfv * self.config.delta_t
        TFVSpd = max(TFVSpd, 0.0)
 
        olv_ego_rel_spd = OLVSpd - ego_prev_speed
        ofv_ego_rel_spd = ofv_prev_speed - ego_prev_speed
        self.OFV_OLV_space += (olv_ego_rel_spd - ofv_ego_rel_spd) * self.config.delta_t
        self.OFV_OLV_space = max(self.OFV_OLV_space, self.idm_params.s0)
        
        tlv_ego_rel_spd = TLVSpd - ego_prev_speed
        tfv_ego_rel_spd = tfv_prev_speed - ego_prev_speed
        self.TFV_TLV_space += (tlv_ego_rel_spd - tfv_ego_rel_spd) * self.config.delta_t
        self.TFV_TLV_space = max(self.TFV_TLV_space, self.idm_params.s0)
 
        OLVRelSpd = LonSpd - OLVSpd
        OFVRelSpd = LonSpd - OFVSpd
        TLVRelSpd = LonSpd - TLVSpd
        TFVRelSpd = LonSpd - TFVSpd
        
        OLVSpace = ego_prev_OLV_space + OLVRelSpd * self.config.delta_t
        OFVSpace = ego_prev_OFV_space + OFVRelSpd * self.config.delta_t
        TLVSpace = ego_prev_TLV_space + TLVRelSpd * self.config.delta_t
        TFVSpace = ego_prev_TFV_space + TFVRelSpd * self.config.delta_t
        
        self.currentState = [
            LonSpd, CrsSpd, CrsLoc,
            OLVSpace, OFVSpace, TLVSpace, TFVSpace,
            OLVRelSpd, OFVRelSpd, TLVRelSpd, TFVRelSpd
        ]
        
        # Update sliding window state vector
        self.s = np.hstack((self.s[11:], self.currentState))[:self.config.timeWindow*11]
        if len(self.s) < self.config.timeWindow*11:
            pad_len = self.config.timeWindow*11 - len(self.s)
            self.s = np.pad(self.s, (0, pad_len), 'constant', constant_values=self.currentState[-1])
        
        # Collision detection
        self.isCollision = 0
        if onOLane:
            if OLVSpace > 1e-6 or OFVSpace < -1e-6:
                self.isCollision = 1
        if onTLane:
            if TLVSpace > 1e-6 or TFVSpace < -1e-6:
                self.isCollision = 1
        fColl = -self.config.Cpenalty * self.isCollision
        
        if self.timeStep - 1 < len(self.OLVSimSpace):
            self.OLVSimSpace[self.timeStep - 1] = OLVSpace
            self.OFVSimSpace[self.timeStep - 1] = OFVSpace
            self.TLVSimSpace[self.timeStep - 1] = TLVSpace
            self.TFVSimSpace[self.timeStep - 1] = TFVSpace
            self.CRSSimSpace[self.timeStep - 1] = CrsLoc
            self.LonSimSpeed[self.timeStep - 1] = LonSpd
            self.CrsSimSpeed[self.timeStep - 1] = CrsSpd
        
        # Reward calculation
        # Jerk reward
        LonJerk = (action[0] - self.lastAction[0]) / self.config.delta_t
        CrsJerk = (action[1] - self.lastAction[1]) / self.config.delta_t
        fLonJerk = 1 / ((LonJerk**2) / 3600 + 1)
        fCrsJerk = 1 / ((CrsJerk**2) / 3600 + 1)
        
        # Acceleration reward
        fLonAcc = 1 / (action[0]**2 / 60 + 1)
        fCrsAcc = 1 / (action[1]**2 / 60 + 1)
        
        # TTC reward
        ttc_gamma = 1
        OLVTTC = -OLVSpace / OLVRelSpd if OLVRelSpd != 0 else float('inf')
        OFVTTC = OFVSpace / OFVRelSpd if OFVRelSpd != 0 else float('inf')
        TLVTTC = -TLVSpace / TLVRelSpd if TLVRelSpd != 0 else float('inf')
        TFVTTC = TFVSpace / TFVRelSpd if TFVRelSpd != 0 else float('inf')
        
        fOLVTTC = np.log(OLVTTC/self.config.TTC_threshold)*ttc_gamma if (onOLane and 0<=OLVTTC<=self.config.TTC_threshold) else 0
        fOFVTTC = np.log(OFVTTC/self.config.TTC_threshold)*ttc_gamma if (onOLane and 0<=OFVTTC<=self.config.TTC_threshold) else 0
        fTLVTTC = np.log(TLVTTC/self.config.TTC_threshold)*ttc_gamma if (onTLane and 0<=TLVTTC<=self.config.TTC_threshold) else 0
        fTFVTTC = np.log(TFVTTC/self.config.TTC_threshold)*ttc_gamma if (onTLane and 0<=TFVTTC<=self.config.TTC_threshold) else 0
        fTTC = (onOLane*(fOLVTTC+fOFVTTC) + onTLane*(fTLVTTC+fTFVTTC)) / (2*max(onOLane+onTLane,1))
        
        # Lane change completion reward
        fLCT = 0
        fFinalCrsV0 = 0
        actual_total_frames = 300
        if CrsLoc >= self.config.MaxCrsLoc:
            self.haveChangedLane += 1
            if self.haveChangedLane == 1:
                actual_total_frames = self.timeStep - self.lane_change_start_frame
                
                if self.fixed_completion_time is not None:
                    mu = self.fixed_completion_time
                    sigma = self.fixed_sigma
                elif self.history_mu is not None and self.history_sigma is not None:
                    mu = self.history_mu
                    sigma = self.history_sigma
                else:
                    mu = 170
                    sigma = 30
                
                mu = max(mu, 10)
                sigma = max(sigma, 5)
                
                gamma_lct = 500000
                fLCT = (np.exp(-(max(actual_total_frames, 0.1) - mu)**2 / (2 * sigma**2)) 
                       / (max(actual_total_frames, 0.1) * sigma * np.sqrt(2 * np.pi))) * gamma_lct + 10
                
                fFinalCrsV0 = (-abs(CrsSpd) * (self.timeStep - (self.lane_change_start_frame + actual_total_frames) + 1)) / 500
        
        # Overtime penalty
        fOvertime = 0
        if self.timeStep >= self.config.MaxChangeLaneTime or self.timeStep >= self.TimeLen-10:
            if CrsLoc <= self.config.MaxCrsLoc:
                fOvertime = -1*(self.timeStep/self.config.MaxChangeLaneTime)*self.config.Tpenalty
        if self.timeStep >= self.TimeLen-3:
            fOvertime -= 0.01
        
        # Rear vehicle deceleration penalty
        ofv_speed_change = OFVSpd - self.OFV_prev_speed
        tfv_speed_change = TFVSpd - self.TFV_prev_speed
        f_ofv_decel = -self.config.decel_penalty_weight * min(ofv_speed_change, 0)
        f_tfv_decel = -self.config.decel_penalty_weight * min(tfv_speed_change, 0)
        f_decel_penalty = f_ofv_decel + f_tfv_decel
 
        self.lastAction = action
        reward = (fLonJerk + fCrsJerk + fLonAcc + fCrsAcc + fTTC + 
                  fLCT + fFinalCrsV0 + fColl + fOvertime + f_decel_penalty)
        
        rewardInfo = np.array([
            fLonJerk, fCrsJerk, fLonAcc, fCrsAcc, fOLVTTC, fOFVTTC,
            fTLVTTC, fTFVTTC, fLCT, fFinalCrsV0, fColl, fOvertime,
            f_ofv_decel, f_tfv_decel, f_decel_penalty
        ], dtype=np.float64)
        
        # Termination condition
        done = (self.timeStep == self.TimeLen) or (self.isCollision == 1) or (fOvertime != 0)

        self.OFV_prev_speed = OFVSpd
        self.TFV_prev_speed = TFVSpd
        self.OFV_current_speed = OFVSpd
        self.TFV_current_speed = TFVSpd
        
        return self.s, reward, done, rewardInfo