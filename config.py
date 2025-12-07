@dataclass
class EnvConfig:
    """Environment configuration container"""
    TTC_threshold: float = 3.001
    MaxCrsLoc: float = 3.49
    MaxChangeLaneTime: int = 250
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