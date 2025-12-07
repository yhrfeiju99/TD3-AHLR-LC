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

def nll_loss(preds, train_data):
    y_true = train_data.get_label()
    n_samples = len(y_true)

    if len(preds) == n_samples:
        mu_init = np.mean(y_true)
        log_sigma2_init = 0.0
        preds = np.concatenate([
            np.full(n_samples, mu_init),
            np.full(n_samples, log_sigma2_init)
        ])

    if len(preds) != 2 * n_samples:
        raise ValueError(f"Invalid prediction dimension! Expected={2*n_samples}, Actual={len(preds)}")

    mu = preds[:n_samples]
    log_sigma2 = preds[n_samples:]
    sigma2 = np.exp(log_sigma2)

    loss = 0.5 * log_sigma2 + (y_true - mu) ** 2 / (2 * sigma2)

    grad_mu = (mu - y_true) / sigma2
    grad_log_sigma2 = 0.5 * (1 - (y_true - mu) ** 2 / sigma2)
    grad = np.concatenate([grad_mu, grad_log_sigma2])

    hess_mu = 1 / sigma2
    hess_log_sigma2 = 0.5 * (y_true - mu) ** 2 / sigma2
    hess = np.concatenate([hess_mu, hess_log_sigma2])
    
    return grad, hess

