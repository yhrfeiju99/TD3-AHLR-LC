import lightgbm as lgb
import numpy as np
import pandas as pd
from utils import *

from typing import Dict, Any, Tuple

def train_mean_variance_models(X_train: pd.DataFrame, t_train: pd.Series, 
                               X_test: pd.DataFrame, t_test: pd.Series) -> Tuple[lgb.Booster, lgb.Booster]:
    def _create_lgb_datasets(X_tr: pd.DataFrame, y_tr: np.ndarray, 
                             X_te: pd.DataFrame, y_te: np.ndarray) -> Tuple[lgb.Dataset, lgb.Dataset]:
        train_set = lgb.Dataset(X_tr, label=y_tr)
        eval_set = lgb.Dataset(X_te, label=y_te, reference=train_set)
        return train_set, eval_set
    
    lgb_train_mu, lgb_eval_mu = _create_lgb_datasets(X_train, t_train.values, X_test, t_test.values)

    params_mu: Dict[str, Any] = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.005,
        'num_leaves': 64,
        'max_depth': 12,
        'verbose': 2,
        'seed': 42,
        'n_jobs': -1,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }
 
    callbacks_mu = [
        lgb.early_stopping(stopping_rounds=150, verbose=True),
        lgb.log_evaluation(period=100)
    ]
 
    model_mu = lgb.train(
        params=params_mu,
        train_set=lgb_train_mu,
        num_boost_round=3000,
        valid_sets=[lgb_eval_mu],
        callbacks=callbacks_mu
    )
 
    y_pred_mu_train = model_mu.predict(X_train, num_iteration=model_mu.best_iteration)
    y_pred_mu_test = model_mu.predict(X_test, num_iteration=model_mu.best_iteration)
 
    residual_sq_train = np.square(t_train.values - y_pred_mu_train)
    residual_sq_test = np.square(t_test.values - y_pred_mu_test)
 
    lgb_train_sigma, lgb_eval_sigma = _create_lgb_datasets(X_train, residual_sq_train, 
                                                           X_test, residual_sq_test)
 
    model_sigma = lgb.train(
        params=params_mu,
        train_set=lgb_train_sigma,
        num_boost_round=3000,
        valid_sets=[lgb_eval_sigma],
        callbacks=callbacks_mu
    )
    
    return model_mu, model_sigma

def generate_predictions(model_mu: lgb.Booster, model_sigma: lgb.Booster, 
                         X_test: pd.DataFrame, t_test: pd.Series) -> pd.DataFrame:
 
    mu_pred = model_mu.predict(X_test, num_iteration=model_mu.best_iteration)
    sigma2_pred = model_sigma.predict(X_test, num_iteration=model_sigma.best_iteration)
 
    sigma2_pred = np.maximum(sigma2_pred, 1e-6)
    sigma_pred = np.sqrt(sigma2_pred)
 
    ci_95_lower = mu_pred - 1.96 * sigma_pred
    ci_95_upper = mu_pred + 1.96 * sigma_pred
 
    result_df = pd.DataFrame({
        'true_remaining_time': t_test.values,
        'pred_mu': mu_pred,
        'pred_sigma': sigma_pred,
        'pred_sigma2': sigma2_pred,
        '95%_CI_lower': ci_95_lower,
        '95%_CI_upper': ci_95_upper
    })
    
    return result_df

# Execute model training and prediction pipeline
if __name__ == "__main__":

    model_mu, model_sigma = train_mean_variance_models(X_train, t_train, X_test, t_test)
 
    final_predictions = generate_predictions(model_mu, model_sigma, X_test, t_test)