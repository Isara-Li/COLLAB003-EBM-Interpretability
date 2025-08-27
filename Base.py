import argparse
import os
import random
import joblib
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

# InterpretML
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
# Optuna for Bayesian-style optimization
import optuna

# PyTorch for self-supervised pretraining
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


##### ---------- Utility: fairness & robustness metrics ---------- #####
def demographic_parity_difference(y_true, y_pred_proba, sensitive: np.ndarray, threshold=0.5):
    """Abs difference between positive rates across groups (binary sensitive)."""
    preds = (y_pred_proba >= threshold).astype(int)
    groups = np.unique(sensitive)
    rates = []
    for g in groups:
        mask = (sensitive == g)
        if mask.sum() == 0:
            rates.append(0.0)
        else:
            rates.append(preds[mask].mean())
    return abs(rates[0] - rates[1]) if len(rates) >= 2 else 0.0


def equalized_odds_difference(y_true, y_pred_proba, sensitive: np.ndarray, threshold=0.5):
    """Max absolute difference in TPR and FPR across groups (binary sensitive)."""
    preds = (y_pred_proba >= threshold).astype(int)
    groups = np.unique(sensitive)
    tprs = []
    fprs = []
    for g in groups:
        mask = (sensitive == g)
        if mask.sum() == 0:
            tprs.append(0.0)
            fprs.append(0.0)
            continue
        tp = ((y_true == 1) & (preds == 1) & mask).sum()
        fn = ((y_true == 1) & (preds == 0) & mask).sum()
        fp = ((y_true == 0) & (preds == 1) & mask).sum()
        tn = ((y_true == 0) & (preds == 0) & mask).sum()
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    # Differences
    tpr_diff = abs(tprs[0] - tprs[1]) if len(tprs) >= 2 else 0.0
    fpr_diff = abs(fprs[0] - fprs[1]) if len(fprs) >= 2 else 0.0
    return max(tpr_diff, fpr_diff)


def empirical_robustness_noise(model_predict_proba, X, y, noise_std_list=[0.01, 0.05, 0.1]):
    """Simple robustness: drop in ROC AUC under additive Gaussian noise."""
    base_auc = roc_auc_score(y, model_predict_proba(X))
    results = {'base_auc': base_auc}
    for std in noise_std_list:
        Xp = X + np.random.normal(0, std, size=X.shape)
        auc = roc_auc_score(y, model_predict_proba(Xp))
        results[f'auc_noise_{std}'] = auc
    return results


##### ---------- Data loaders for UCI-like datasets ---------- #####
def load_adult_dataset() -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load Adult from openml via sklearn fetch_openml (or fallback to local if necessary)."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.copy()
    # label mapping to binary: >50K -> 1
    df = df.rename(columns={'class': 'income'})
    df['income'] = (df['income'].apply(lambda s: 1 if '>50K' in s else 0)).astype(int)
    y = df['income']
    X = df.drop(columns=['income'])
    # choose 'sex' as sensitive attribute for fairness experiments
    sensitive = (X['sex'] == 'Male').astype(int).values
    return X, y, sensitive


def load_breast_cancer_sklearn():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    X = d.frame.drop(columns=['target'])
    y = d.target
    # synthetic sensitive attribute (for demo) - use mean of features as split
    sensitive = (X.iloc[:, 0] > X.iloc[:, 0].median()).astype(int).values
    return X, y, sensitive


DATASET_LOADERS = {
    'adult': load_adult_dataset,
    'breast': load_breast_cancer_sklearn,
}


##### ---------- Preprocessing pipeline ---------- #####
def build_preprocessor(X: pd.DataFrame):
    """Auto-detect categorical and numeric columns and build a ColumnTransformer."""
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number, 'float64', 'int64']).columns.tolist()
    # Simple encoding for categorical features
    transformers = []
    if len(num_cols):
        transformers.append(('num', StandardScaler(), num_cols))
    if len(cat_cols):
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
    preproc = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)
    return preproc


##### ---------- Baseline EBM trainer ---------- #####
def train_baseline_ebm(X_train, y_train, X_val, y_val, params=None, init_score=None, sample_weight=None):
    params = params or {}
    model = ExplainableBoostingClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight, init_score=init_score)
    preds_proba = model.predict_proba(X_val)[:, 1]
    metrics = {
        'roc_auc': roc_auc_score(y_val, preds_proba),
        'f1': f1_score(y_val, (preds_proba >= 0.5).astype(int)),
        'accuracy': accuracy_score(y_val, (preds_proba >= 0.5).astype(int))
    }
    return model, metrics, preds_proba


##### ---------- Optuna HPO: composite objective (accuracy + fairness) ---------- #####
def run_hpo_ebm(X_train, y_train, X_val, y_val, sensitive_val, n_trials=40, timeout=None, output_dir='hpo'):
    os.makedirs(output_dir, exist_ok=True)

    def objective(trial):
        # sample hyperparams
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            'max_bins': trial.suggest_int('max_bins', 64, 512),
            'max_leaves': trial.suggest_int('max_leaves', 2, 64),
            'max_rounds': trial.suggest_int('max_rounds', 50, 2000),
            'interactions': trial.suggest_int('interactions', 0, 10),
            'outer_bags': trial.suggest_int('outer_bags', 4, 32),
            'inner_bags': trial.suggest_int('inner_bags', 0, 8),
            # greedy ratio can be tuned too
            'greedy_ratio': trial.suggest_float('greedy_ratio', 0.0, 20.0),
        }
        # a tradeoff weight alpha between accuracy and fairness to optimize externally
        fairness_weight = trial.suggest_float('fairness_weight', 0.0, 5.0)

        model = ExplainableBoostingClassifier(**params)
        model.fit(X_train, y_train)
        preds_proba = model.predict_proba(X_val)[:, 1]
        roc = roc_auc_score(y_val, preds_proba)
        dp = demographic_parity_difference(y_val.values, preds_proba, sensitive_val, threshold=0.5)
        # composite: minimize (1 - roc) + fairness_weight * dp
        objective_value = (1.0 - roc) + fairness_weight * dp

        # store additional info as trial user attributes
        trial.set_user_attr('roc_auc', roc)
        trial.set_user_attr('dp', dp)
        trial.set_user_attr('params', params)
        return objective_value

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    # persist study
    joblib.dump(study, os.path.join(output_dir, 'optuna_study.pkl'))
    return study


##### ---------- Self-supervised pretraining (autoencoder) ---------- #####
class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(64, latent_dim * 4)),
            nn.ReLU(),
            nn.Linear(max(64, latent_dim * 4), latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(64, latent_dim * 4)),
            nn.ReLU(),
            nn.Linear(max(64, latent_dim * 4), input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def pretrain_autoencoder_get_init_scores(X_unlabeled: np.ndarray, X_small_labeled: np.ndarray, y_small: np.ndarray,
                                         device='cpu', epochs=50, batch_size=256, latent_dim=32):
    """
    Train autoencoder on X_unlabeled (self-supervised), then train a small classifier on
    the encoder output for the labeled subset to produce init_score probabilities.
    Returns init_score for entire dataset (probabilities).
    """
    device = torch.device(device)
    X_unlabeled_t = torch.from_numpy(X_unlabeled.astype(np.float32))
    ds = TensorDataset(X_unlabeled_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    ae = TabularAutoencoder(input_dim=X_unlabeled.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        ae.train()
        epoch_loss = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            recon, _ = ae(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
    # build encoder outputs for labeled small set
    ae.eval()
    with torch.no_grad():
        Z_small = ae.encoder(torch.from_numpy(X_small_labeled.astype(np.float32)).to(device)).cpu().numpy()
        Z_full = ae.encoder(torch.from_numpy(X_unlabeled.astype(np.float32)).to(device)).cpu().numpy()

    # train logistic regression on Z_small -> produce probabilities for full set as init_score
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Z_small, y_small)
    init_scores = clf.predict_proba(Z_full)[:, 1]
    return init_scores, ae, clf


##### ---------- Orchestrator / main experiment flow ---------- #####
def run_pipeline(dataset='adult', do_hpo=True, do_pretrain=True, output_dir='out'):
    os.makedirs(output_dir, exist_ok=True)
    print(f'[INFO] Loading {dataset}')
    X_df, y, sensitive = DATASET_LOADERS[dataset]()
    preproc = build_preprocessor(X_df)
    X_all = preproc.fit_transform(X_df)
    # convert to numpy arrays for later use
    X_all = np.asarray(X_all, dtype=float)
    y = np.asarray(y, dtype=int)

    # train/val/test split
    X_temp, X_test, y_temp, y_test, sens_temp, sens_test = train_test_split(
        X_all, y, sensitive, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val, sens_train, sens_val = train_test_split(
        X_temp, y_temp, sens_temp, test_size=0.25, stratify=y_temp, random_state=SEED)  # 0.25*0.8 = 0.2

    # Baseline EBM
    print('[INFO] Training baseline EBM (default params)...')
    ebm_params = {}  # defaults
    ebm, base_metrics, base_preds_proba = train_baseline_ebm(X_train, y_train, X_val, y_val, params=ebm_params)
    print('[BASELINE]', base_metrics)

    # Save baseline model
    joblib.dump(ebm, os.path.join(output_dir, 'ebm_baseline.pkl'))

    # HPO
    study = None
    if do_hpo:
        print('[INFO] Running HPO (Optuna)...')
        study = run_hpo_ebm(X_train, y_train, X_val, y_val, sens_val, n_trials=30, output_dir=output_dir)
        print('[HPO] Best trial:', study.best_trial.params)
        # Train model with best params and evaluate on validation and test
        best_params = study.best_trial.user_attrs.get('params', {})
        ebm_best, best_metrics, best_preds = train_baseline_ebm(X_train, y_train, X_val, y_val, params=best_params)
        print('[BEST_ON_VAL]', best_metrics)
        joblib.dump(ebm_best, os.path.join(output_dir, 'ebm_best_val.pkl'))

    # Self-supervised pretraining -> init_score
    if do_pretrain:
        print('[INFO] Pretraining autoencoder and deriving init_score...')
        # Use all X_train + X_val + X_test as unlabeled pool for pretraining; use small labeled subset for classifier on latent.
        X_unlabeled = X_all  # full data for self-supervision (practical choice)
        # small labeled subset (simulate cold-start)
        small_idx = np.random.choice(range(len(X_train)), size=max(20, int(0.05 * len(X_train))), replace=False)
        X_small_labeled = X_train[small_idx]
        y_small = y_train[small_idx]
        init_scores_full, ae_model, small_clf = pretrain_autoencoder_get_init_scores(
            X_unlabeled, X_small_labeled, y_small, epochs=20, batch_size=256, latent_dim=32)
        # Train EBM with init_score (init_score accepts per-sample scores)
        # NOTE: InterpretML expects init_score to be either per-sample score array or a model that produces scores.
        print('[INFO] Training EBM initialized from pretrain init_scores...')
        ebm_pre, pre_metrics, pre_preds = train_baseline_ebm(X_train, y_train, X_val, y_val, init_score=init_scores_full[:len(X_train)])
        print('[PRETRAIN_INIT]', pre_metrics)
        joblib.dump(ebm_pre, os.path.join(output_dir, 'ebm_pretrained_init.pkl'))

    # Final evaluation on holdout test set for chosen best model (prefer HPO best if available)
    final_model = ebm
    if study is not None:
        # load best model trained earlier if exists
        try:
            final_model = joblib.load(os.path.join(output_dir, 'ebm_best_val.pkl'))
        except Exception:
            final_model = ebm
    print('[INFO] Final evaluation on test set')
    preds_test_proba = final_model.predict_proba(X_test)[:, 1]
    test_metrics = {
        'roc_auc': roc_auc_score(y_test, preds_test_proba),
        'f1': f1_score(y_test, (preds_test_proba >= 0.5).astype(int)),
        'accuracy': accuracy_score(y_test, (preds_test_proba >= 0.5).astype(int))
    }
    dp_test = demographic_parity_difference(y_test, preds_test_proba, sens_test)
    eo_test = equalized_odds_difference(y_test, preds_test_proba, sens_test)
    robustness = empirical_robustness_noise(lambda Xq: final_model.predict_proba(Xq)[:, 1], X_test, y_test)

    results = {
        'test_metrics': test_metrics,
        'demographic_parity': dp_test,
        'equalized_odds': eo_test,
        'robustness': robustness
    }
    print(results)
    # persist results
    joblib.dump(results, os.path.join(output_dir, 'final_results.pkl'))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult', choices=list(DATASET_LOADERS.keys()))
    parser.add_argument('--run-hpo', dest='run_hpo', action='store_true')
    parser.add_argument('--use-pretrain', dest='use_pretrain', action='store_true')
    parser.add_argument('--out', type=str, default='out')
    args, unknown = parser.parse_known_args()
    run_pipeline(dataset=args.dataset, do_hpo=args.run_hpo, do_pretrain=args.use_pretrain, output_dir=args.out)
