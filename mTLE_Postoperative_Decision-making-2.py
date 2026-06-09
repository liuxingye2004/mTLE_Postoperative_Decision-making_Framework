#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import chi2, norm, binom

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 24,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.weight': 'bold',
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'figure.dpi': 300
})

from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

MODEL_FOLDER = None
EXTERNAL_FOLDER = None
EXTERNAL_FILES = None
TARGET_COL = 'FUFA'
RANDOM_STATE = 42

class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.pls = PLSRegression(n_components=n_components)
        self.calibrator_ = None
        self.threshold = 0.5

    def fit(self, X, y):
        self.pls.fit(X, y)
        y_cont = self.pls.predict(X).flatten()
        self.calibrator_ = LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=1000, random_state=self.random_state
        )
        self.calibrator_.fit(y_cont.reshape(-1, 1), y)
        self.threshold = 0.5
        return self

    def predict_proba(self, X):
        y_cont = self.pls.predict(X).flatten()
        proba = self.calibrator_.predict_proba(y_cont.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

class MultinomialNBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.scaler_ = MinMaxScaler(feature_range=(0, 1))
        self.model_ = MultinomialNB(alpha=self.alpha)
        self.classes_ = None
    def fit(self, X, y):
        X_nn = self.scaler_.fit_transform(X)
        self.model_.fit(X_nn, y)
        self.classes_ = self.model_.classes_
        return self
    def predict(self, X):
        X_nn = self.scaler_.transform(X)
        return self.model_.predict(X_nn)
    def predict_proba(self, X):
        X_nn = self.scaler_.transform(X)
        return self.model_.predict_proba(X_nn)

class BernoulliNBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.scaler_ = MinMaxScaler(feature_range=(0, 1))
        self.model_ = BernoulliNB(alpha=self.alpha, binarize=0.5)
        self.classes_ = None
    def fit(self, X, y):
        X_nn = self.scaler_.fit_transform(X)
        self.model_.fit(X_nn, y)
        self.classes_ = self.model_.classes_
        return self
    def predict(self, X):
        X_nn = self.scaler_.transform(X)
        return self.model_.predict(X_nn)
    def predict_proba(self, X):
        X_nn = self.scaler_.transform(X)
        return self.model_.predict_proba(X_nn)

class StackingWrapper:
    def __init__(self, base_models, meta_model, feature_names):
        self.base_models = base_models
        self.meta_model = meta_model
        self.feature_names = feature_names
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names)
        meta_dict = {}
        for name, model in self.base_models.items():
            try:
                meta_dict[name] = model.predict_proba(X_df)[:, 1]
            except Exception:
                meta_dict[name] = np.zeros(len(X_df)) + 0.5
        X_meta = pd.DataFrame(meta_dict)
        return self.meta_model.predict_proba(X_meta)
    def predict(self, X):
        proba = self.predict_proba(X)
        return proba[:, 1] if proba.ndim > 1 else proba

TOP5_COLORS = ['#D62828', '#F77F00', '#FCBF49', '#06A77D', '#118AB2']

def auto_detect_model_folder():
    model_folders = glob.glob('*_输出模型')
    if not model_folders:
        raise FileNotFoundError("未检测到模型文件夹（*_输出模型），请先运行代码包3")
    model_folder = sorted(model_folders)[-1]
    print(f"[INFO] 模型文件夹: {model_folder}")
    return model_folder

def auto_detect_external_files():
    if EXTERNAL_FILES is not None:
        return EXTERNAL_FILES

    if EXTERNAL_FOLDER and os.path.isdir(EXTERNAL_FOLDER):
        search_path = EXTERNAL_FOLDER
        print(f"[INFO] 外部数据: {search_path}")
    else:
        default_path = './output_external_imputation'
        if os.path.isdir(default_path):
            search_path = default_path
            print(f"[INFO] 外部数据: {search_path}")
        else:
            search_path = '.'
            print(f"[WARN] 回退到当前路径: {search_path}")

    all_files = sorted(
        glob.glob(os.path.join(search_path, '*.xlsx')) +
        glob.glob(os.path.join(search_path, '*.xls')) +
        glob.glob(os.path.join(search_path, '*.csv'))
    )
    all_files = [f for f in all_files if not os.path.basename(f).startswith('~')]

    seen_stems = {}
    priority = {'.xlsx': 3, '.xls': 2, '.csv': 1}
    for f in all_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        ext = os.path.splitext(f)[1].lower()
        if stem not in seen_stems:
            seen_stems[stem] = f
        else:
            current_ext = os.path.splitext(seen_stems[stem])[1].lower()
            if priority.get(ext, 0) > priority.get(current_ext, 0):
                seen_stems[stem] = f
    all_files = sorted(seen_stems.values())

    exclude_keywords = ['train', 'selected', 'full', 'training', 'internal', 'data', 'output', 'imputation', 'feature']
    ext_files = [f for f in all_files if not any(k in os.path.basename(f).lower() for k in exclude_keywords)]
    if not ext_files:
        ext_files = all_files
    if not ext_files:
        raise FileNotFoundError(f"未在 {search_path} 检测到外部验证文件")
    print(f"[INFO] 检测到 {len(ext_files)} 个外部队列:")
    for f in ext_files:
        print(f"  - {f}")
    return ext_files

def smart_read_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.xlsx', '.xlsm', '.xlsb']:
        try:
            df = pd.read_excel(filepath, engine='openpyxl')
            print(f"  [INFO] 读取: {os.path.basename(filepath)}")
            return df
        except Exception:
            pass

    if ext in ['.xls', '.xlsx']:
        try:
            df = pd.read_excel(filepath, engine='xlrd')
            print(f"  [INFO] 读取: {os.path.basename(filepath)}")
            return df
        except Exception:
            pass

    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'cp1252']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"  [INFO] 读取: {os.path.basename(filepath)} ({enc})")
            return df
        except UnicodeDecodeError:
            continue
        except Exception:
            break

    raise ValueError(f"无法读取文件 {filepath}")

def save_df(df, filepath, float_fmt='%.3f'):
    try:
        df.to_csv(filepath, index=False, encoding='utf-8-sig', float_format=float_fmt)
        print(f"[INFO] 已保存: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"[ERR] 保存失败 {os.path.basename(filepath)}: {e}")

def compute_metrics_with_threshold(y_true, y_pred_prob, threshold=None):
    try:
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_prob_arr = np.asarray(y_pred_prob).ravel()
        auc_val = roc_auc_score(y_true_arr, y_pred_prob_arr)
        pr_auc_val = average_precision_score(y_true_arr, y_pred_prob_arr)
        if threshold is None:
            fpr, tpr, thresholds = roc_curve(y_true_arr, y_pred_prob_arr)
            best_idx = np.argmax(tpr - fpr) if len(thresholds) > 0 else 0
            threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
        y_pred = (y_pred_prob_arr >= threshold).astype(int)
        cm = confusion_matrix(y_true_arr, y_pred, labels=[0,1])
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = np.sum((y_true_arr==0)&(y_pred==0)); fp = np.sum((y_true_arr==0)&(y_pred==1))
            fn = np.sum((y_true_arr==1)&(y_pred==0)); tp = np.sum((y_true_arr==1)&(y_pred==1))
        sensitivity = tp/(tp+fn) if (tp+fn)>0 else 0
        specificity = tn/(tn+fp) if (tn+fp)>0 else 0
        ppv = tp/(tp+fp) if (tp+fp)>0 else 0
        npv = tn/(tn+fn) if (tn+fn)>0 else 0
        accuracy = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0
        f1 = 2*ppv*sensitivity/(ppv+sensitivity) if (ppv+sensitivity)>0 else 0
        g_mean = np.sqrt(sensitivity*specificity)
        return {
            'AUC': round(auc_val,3), 'PR_AUC': round(pr_auc_val,3),
            'Accuracy': round(accuracy,3), 'F1_score': round(f1,3),
            'NPV': round(npv,3), 'PPV': round(ppv,3),
            'Sensitivity': round(sensitivity,3), 'Specificity': round(specificity,3),
            'G_mean': round(g_mean,3), 'Threshold': round(threshold,3)
        }, float(threshold)
    except Exception:
        return {'AUC':0,'PR_AUC':0,'Accuracy':0,'F1_score':0,'NPV':0,'PPV':0,
                'Sensitivity':0,'Specificity':0,'G_mean':0,'Threshold':0.5}, 0.5

def brier_score(y_true, y_pred_prob):
    return round(np.mean((np.asarray(y_true) - np.asarray(y_pred_prob))**2), 3)


def bootstrap_pr_auc_ci(y_true, y_pred_prob, n_bootstrap=1000, alpha=0.05):
    try:
        np.random.seed(42)
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred_prob)
        n = len(y_true_arr)
        theta_hat = average_precision_score(y_true_arr, y_pred_arr)
        pr_aucs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            if len(np.unique(y_true_arr[idx])) == 2:
                pr_aucs.append(average_precision_score(y_true_arr[idx], y_pred_arr[idx]))
        if len(pr_aucs) < 100:
            return f"{theta_hat:.3f} (NA)", theta_hat, np.nan, np.nan
        pr_aucs = np.array(pr_aucs)
        z0 = norm.ppf(np.mean(pr_aucs < theta_hat))
        lower = np.percentile(pr_aucs, norm.cdf(z0 + (z0 + norm.ppf(alpha/2))) * 100)
        upper = np.percentile(pr_aucs, norm.cdf(z0 + (z0 + norm.ppf(1-alpha/2))) * 100)
        lower, upper = max(0.0, min(1.0, lower)), max(0.0, min(1.0, upper))
        return f"{theta_hat:.3f} ({lower:.3f}-{upper:.3f})", theta_hat, lower, upper
    except Exception:
        ap = average_precision_score(y_true, y_pred_prob)
        return f"{ap:.3f} (Error)", ap, 0.0, 0.0

def _loess_smooth(x, y, frac=0.4):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 4:
        return x, y
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        order = np.argsort(x)
        x_ord, y_ord = x[order], y[order]
        smoothed = lowess(y_ord, x_ord, frac=frac, it=1, return_sorted=True)
        return smoothed[:, 0], smoothed[:, 1]
    except Exception:
        window = max(2, len(x) // 3)
        y_smooth = np.convolve(y, np.ones(window)/window, mode='same')
        return x, y_smooth

def _quantile_calibration_bins(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n = len(y_true)
    if n_bins is None:
        n_bins = min(10, max(5, n // 25))
    if n < n_bins * 2:
        n_bins = max(3, n // 2)
    order = np.argsort(y_prob)
    y_prob_sorted = y_prob[order]
    y_true_sorted = y_true[order]
    bin_size = n // n_bins
    prob_pred_list, prob_true_list, counts_list = [], [], []
    for i in range(n_bins):
        start = i * bin_size
        end = n if i == n_bins - 1 else start + bin_size
        if start >= n:
            break
        subset_true = y_true_sorted[start:end]
        subset_prob = y_prob_sorted[start:end]
        count = len(subset_prob)
        if count > 0:
            prob_pred_list.append(float(np.mean(subset_prob)))
            prob_true_list.append(float(np.mean(subset_true)))
            counts_list.append(count)
    return np.array(prob_pred_list), np.array(prob_true_list), np.array(counts_list)

def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    brier = float(brier_score(y_true, y_prob))
    return {'Brier': round(brier, 3)}

def bootstrap_auc_ci(y_true, y_pred_prob, n_bootstrap=1000, alpha=0.05):
    try:
        np.random.seed(42); y_true_arr = np.asarray(y_true); y_pred_arr = np.asarray(y_pred_prob); n = len(y_true_arr)
        theta_hat = roc_auc_score(y_true_arr, y_pred_arr)
        aucs = [roc_auc_score(y_true_arr[idx], y_pred_arr[idx]) 
                for _ in range(n_bootstrap) 
                if len(np.unique((idx:=np.random.choice(n,n,replace=True))))>1 and len(np.unique(y_true_arr[idx]))==2]
        if len(aucs) < 100: return f"{theta_hat:.3f} (NA)", theta_hat, np.nan, np.nan
        aucs = np.array(aucs)
        z0 = norm.ppf(np.mean(aucs < theta_hat))
        lower = np.percentile(aucs, norm.cdf(z0+(z0+norm.ppf(alpha/2)))*100)
        upper = np.percentile(aucs, norm.cdf(z0+(z0+norm.ppf(1-alpha/2)))*100)
        lower, upper = max(0.0, min(1.0, lower)), max(0.0, min(1.0, upper))
        return f"{theta_hat:.3f} ({lower:.3f}-{upper:.3f})", theta_hat, lower, upper
    except Exception:
        return f"{roc_auc_score(y_true, y_pred_prob):.3f} (Error)", roc_auc_score(y_true, y_pred_prob), 0.5, 0.5

def delong_test(y_true, y_pred_prob1, y_pred_prob2):
    y_true = np.asarray(y_true); s1 = np.asarray(y_pred_prob1); s2 = np.asarray(y_pred_prob2)
    auc1 = roc_auc_score(y_true, s1); auc2 = roc_auc_score(y_true, s2)
    pos_idx, neg_idx = np.where(y_true==1)[0], np.where(y_true==0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos==0 or n_neg==0: return 0.0, 1.0, round(auc1,3), round(auc2,3)
    def v10(scores, idx):
        return np.array([np.mean(scores[neg_idx] < scores[i]) + 0.5*np.mean(scores[neg_idx]==scores[i]) for i in idx])
    def v01(scores, idx):
        return np.array([np.mean(scores[pos_idx] > scores[i]) + 0.5*np.mean(scores[pos_idx]==scores[i]) for i in idx])
    V10_1, V10_2 = v10(s1, pos_idx), v10(s2, pos_idx)
    V01_1, V01_2 = v01(s1, neg_idx), v01(s2, neg_idx)
    var1 = (np.var(V01_1, ddof=1) if n_neg>1 else 0)/n_neg + (np.var(V10_1, ddof=1) if n_pos>1 else 0)/n_pos
    var2 = (np.var(V01_2, ddof=1) if n_neg>1 else 0)/n_neg + (np.var(V10_2, ddof=1) if n_pos>1 else 0)/n_pos
    cov = (np.cov(V01_1, V01_2, ddof=1)[0,1] if n_neg>1 else 0)/n_neg + (np.cov(V10_1, V10_2, ddof=1)[0,1] if n_pos>1 else 0)/n_pos
    var_diff = var1 + var2 - 2*cov
    if var_diff <= 0: var_diff = 1e-10
    z = (auc1-auc2)/np.sqrt(var_diff)
    return round(z,3), round(2*(1-norm.cdf(abs(z))),3), round(auc1,3), round(auc2,3)

def adjust_pvalues(df, p_col='P', method='fdr_bh'):
    if p_col not in df.columns: return df
    pvals = df[p_col].astype(float).values; n_tests = len(pvals)
    if STATSMODELS_AVAILABLE:
        try:
            reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method=method, is_sorted=False, returnsorted=False)
            df[f'{p_col}_FDR'] = np.round(p_adj, 3); df['Significant_FDR'] = reject
        except Exception:
            df[f'{p_col}_Bonferroni'] = np.round(np.minimum(pvals*n_tests, 1.0), 3)
            df['Significant_Bonf'] = df[f'{p_col}_Bonferroni'] < 0.05
    else:
        df[f'{p_col}_Bonferroni'] = np.round(np.minimum(pvals*n_tests, 1.0), 3)
        df['Significant_Bonf'] = df[f'{p_col}_Bonferroni'] < 0.05
    return df

def comprehensive_ranking_with_ranks(df, performance_metrics, rank_col_name='Comprehensive_Rank'):
    rank_cols = []
    for metric in performance_metrics:
        if metric in df.columns:
            rank_col = f'{metric}_rank'
            df[rank_col] = df[metric].rank(ascending=False, method='min').astype(int)
            rank_cols.append(rank_col)
    if rank_cols:
        df[rank_col_name] = df[rank_cols].mean(axis=1).round(3)
    df = df.sort_values(rank_col_name)
    return df, rank_cols

def _set_bold_ticks(ax, labelsize=24):
    ax.tick_params(axis='both', labelsize=labelsize, width=2.5, length=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

def _set_spines(ax, lw=2.5):
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
        spine.set_color('black')

def _auto_resize_legend(ax, legend, min_fontsize=10):
    """自动检测图例是否超出axes边界，超出则逐步缩小字体"""
    fig = ax.figure
    fig.canvas.draw()
    ax_bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    leg_bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    margin = 0.02
    ax_left, ax_bottom = ax_bbox.x0, ax_bbox.y0
    ax_right, ax_top = ax_bbox.x1, ax_bbox.y1
    leg_left, leg_bottom = leg_bbox.x0, leg_bbox.y0
    leg_right, leg_top = leg_bbox.x1, leg_bbox.y1
    while True:
        if (leg_left >= ax_left - margin and leg_bottom >= ax_bottom - margin and
            leg_right <= ax_right + margin and leg_top <= ax_top + margin):
            break
        current_size = legend.get_texts()[0].get_fontsize() if legend.get_texts() else 24
        if current_size <= min_fontsize:
            break
        new_size = current_size - 2
        for text in legend.get_texts():
            text.set_fontsize(new_size)
        fig.canvas.draw()
        leg_bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        leg_left, leg_bottom = leg_bbox.x0, leg_bbox.y0
        leg_right, leg_top = leg_bbox.x1, leg_bbox.y1
    legend.set_zorder(1)

def plot_roc_clean(fpr, tpr, color, model_name, output_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=3, label=f'{model_name} (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Reference')
    ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=28, fontweight='bold')
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=28, fontweight='bold')
    ax.set_title(f'ROC Curve - {model_name}', fontsize=30, fontweight='bold', pad=15)
    _set_bold_ticks(ax, 24)
    legend = ax.legend(loc='lower right', fontsize=28, frameon=False, labelspacing=0.4, handlelength=1.5)
    for text in legend.get_texts():
        text.set_fontweight('normal')
    _auto_resize_legend(ax, legend)
    ax.grid(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    _set_spines(ax, 2.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def plot_pr_clean(precision, recall, color, model_name, output_path, y_true=None, y_prob=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    ap_val = average_precision_score(y_true, y_prob) if y_prob is not None and y_true is not None else 0
    if y_true is not None:
        baseline = np.mean(y_true)
        ax.axhline(y=baseline, color='gray', linestyle='--', lw=2, alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.plot(recall, precision, color=color, lw=3, label=f'{model_name} ({ap_val:.3f})')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=28, fontweight='bold')
    ax.set_ylabel('Precision (PPV)', fontsize=28, fontweight='bold')
    ax.set_title(f'PR Curve - {model_name}', fontsize=30, fontweight='bold', pad=15)
    _set_bold_ticks(ax, 24)
    legend = ax.legend(loc='lower left', fontsize=28, frameon=False, labelspacing=0.4, handlelength=1.5)
    for text in legend.get_texts():
        text.set_fontweight('normal')
    _auto_resize_legend(ax, legend)
    ax.grid(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    _set_spines(ax, 2.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def plot_dca_standard(y_true, y_pred_prob, color, model_name, output_path):
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        thresholds = np.linspace(0.01, 0.99, 200)
        y_true_arr = np.asarray(y_true)
        n = len(y_true_arr)
        event_rate = np.mean(y_true_arr)
        net_benefit = []
        for thresh in thresholds:
            y_pred = (y_pred_prob >= thresh).astype(int)
            tp = np.sum((y_pred==1) & (y_true_arr==1))
            fp = np.sum((y_pred==1) & (y_true_arr==0))
            if thresh >= 1:
                nb = 0
            else:
                nb = (tp/n) - (fp/n)*(thresh/(1-thresh))
            net_benefit.append(nb)
        net_benefit = np.array(net_benefit)
        treat_all = np.array([(event_rate - (1-event_rate)*(t/(1-t))) if t < 1 else 0 for t in thresholds])
        treat_none = np.zeros_like(thresholds)
        ax.plot(thresholds, net_benefit, color=color, lw=3, label=f'{model_name}')
        ax.plot(thresholds, treat_all, 'k--', lw=2, alpha=0.7, label='Treat All')
        ax.plot(thresholds, treat_none, 'k:', lw=2, alpha=0.7, label='Treat None')
        ax.fill_between(thresholds, 0, net_benefit, where=(net_benefit > 0), alpha=0.2, color=color, interpolate=True)
        ax.set_xlabel('Threshold Probability', fontsize=28, fontweight='bold')
        ax.set_ylabel('Net Benefit', fontsize=28, fontweight='bold')
        ax.set_title(f'Decision Curve Analysis - {model_name}', fontsize=30, fontweight='bold', pad=15)
        _set_bold_ticks(ax, 24)
        legend = ax.legend(loc='upper right', fontsize=28, frameon=False, labelspacing=0.4, handlelength=1.5)
        for text in legend.get_texts():
            text.set_fontweight('normal')
            _auto_resize_legend(ax, legend)
        ax.grid(False)
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.1, 0.25])
        _set_spines(ax, 2.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
        plt.close()
    except Exception as e:
        print(f"DCA曲线绘制失败: {str(e)[:50]}")

def plot_calibration_fixed(y_true, y_prob, color, model_name, output_path, n_bins=None):
    try:
        if n_bins is None:
            n_bins = min(10, max(5, len(y_true) // 25))
        prob_pred, prob_true, counts = _quantile_calibration_bins(y_true, y_prob, n_bins)
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7, label='Ideal')
        if len(prob_pred) < 2:
            ax.plot(prob_pred, prob_true, 's', color=color, markersize=12,
                    markeredgecolor='white', markeredgewidth=2, label=model_name)
        else:
            sort_idx = np.argsort(prob_pred)
            x_pts = prob_pred[sort_idx]
            y_pts = prob_true[sort_idx]
            c_pts = counts[sort_idx]
            for i in range(len(x_pts) - 1):
                x_seg = np.array([x_pts[i], x_pts[i+1]])
                y_seg = np.array([y_pts[i], y_pts[i+1]])
                if c_pts[i] < 5 or c_pts[i+1] < 5:
                    ax.plot(x_seg, y_seg, '--', color='gray', lw=1.5, alpha=0.6, zorder=1)
                else:
                    ax.plot(x_seg, y_seg, '-', color=color, lw=2.5, alpha=0.85, zorder=2)
            sparse_label_added = False
            dense_label_added = False
            for i in range(len(x_pts)):
                if c_pts[i] < 5:
                    label = 'Sparse bin (n<<5)' if not sparse_label_added else ""
                    sparse_label_added = True
                    ax.plot(x_pts[i], y_pts[i], 'o', color='gray', markersize=8,
                            markeredgecolor='white', markeredgewidth=1.0, zorder=3, label=label)
                else:
                    label = model_name if not dense_label_added else ""
                    dense_label_added = True
                    ax.plot(x_pts[i], y_pts[i], 's', color=color, markersize=10,
                            markeredgecolor='white', markeredgewidth=1.5, zorder=3, label=label)
            try:
                x_smooth, y_smooth = _loess_smooth(x_pts, y_pts)
                ax.plot(x_smooth, y_smooth, color=color, lw=1.8, ls='--', alpha=0.55, zorder=4)
            except Exception:
                pass
        ax.set_xlabel('Mean Predicted Probability', fontsize=28, fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontsize=28, fontweight='bold')
        ax.set_title(f'Calibration Curve - {model_name}', fontsize=30, fontweight='bold', pad=15)
        _set_bold_ticks(ax, 24)
        legend = ax.legend(loc='upper left', fontsize=28, frameon=False, labelspacing=0.4, handlelength=1.5)
        for text in legend.get_texts():
            text.set_fontweight('normal')
            _auto_resize_legend(ax, legend)
        ax.grid(False)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        _set_spines(ax, 2.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
        plt.close()
    except Exception as e:
        print(f"校准曲线绘制失败: {str(e)[:50]}")

def plot_combined_roc(models_data, output_path, title):
    fig, ax = plt.subplots(figsize=(16, 12))
    for name, fpr, tpr, color in models_data:
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} ({auc(fpr,tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Reference')
    ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=27, fontweight='bold')
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=27, fontweight='bold')
    ax.set_title(title, fontsize=33, fontweight='bold', pad=15)
    _set_bold_ticks(ax, 24)
    legend = ax.legend(loc='lower right', fontsize=28, frameon=False,
                       ncol=1, labelspacing=0.4, handlelength=1.5)
    for text in legend.get_texts():
        text.set_fontweight('normal')
    _auto_resize_legend(ax, legend)
    ax.grid(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    _set_spines(ax, 2.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def plot_combined_pr(models_data, output_path, title, y_true):
    fig, ax = plt.subplots(figsize=(16, 12))
    for name, y_prob, color in models_data:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color=color, lw=2.5, label=f'{name} ({ap_val:.3f})')
    baseline = np.mean(y_true)
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=2, alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=27, fontweight='bold')
    ax.set_ylabel('Precision (PPV)', fontsize=27, fontweight='bold')
    ax.set_title(title, fontsize=33, fontweight='bold', pad=15)
    _set_bold_ticks(ax, 24)
    legend = ax.legend(loc='lower left', fontsize=28, frameon=False,
                       ncol=1, labelspacing=0.4, handlelength=1.5)
    for text in legend.get_texts():
        text.set_fontweight('normal')
    _auto_resize_legend(ax, legend)
    ax.grid(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    _set_spines(ax, 2.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def plot_combined_calibration(y_true, models_data, output_path, title, n_bins=None):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7, label='Ideal')
    if n_bins is None:
        n_bins = min(10, max(5, len(y_true) // 25))
    for name, y_prob, color in models_data:
        prob_pred, prob_true, counts = _quantile_calibration_bins(y_true, y_prob, n_bins)
        if len(prob_pred) < 2:
            continue
        sort_idx = np.argsort(prob_pred)
        x_pts = prob_pred[sort_idx]
        y_pts = prob_true[sort_idx]
        c_pts = counts[sort_idx]
        for i in range(len(x_pts) - 1):
            x_seg = np.array([x_pts[i], x_pts[i+1]])
            y_seg = np.array([y_pts[i], y_pts[i+1]])
            if c_pts[i] < 5 or c_pts[i+1] < 5:
                ax.plot(x_seg, y_seg, '--', color='gray', lw=1.2, alpha=0.4, zorder=1)
            else:
                ax.plot(x_seg, y_seg, '-', color=color, lw=2.0, alpha=0.8, zorder=2)
        ax.plot(x_pts, y_pts, 's', color=color, markersize=8,
                markeredgecolor='white', markeredgewidth=1.2, zorder=3, label=name)
        try:
            x_smooth, y_smooth = _loess_smooth(x_pts, y_pts)
            ax.plot(x_smooth, y_smooth, color=color, lw=1.5, ls='--', alpha=0.5, zorder=4)
        except Exception:
            pass
    ax.set_xlabel('Mean Predicted Probability', fontsize=27, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=27, fontweight='bold')
    ax.set_title(title, fontsize=33, fontweight='bold', pad=15)
    legend = ax.legend(loc='upper left', fontsize=28, frameon=False,
                       ncol=1, labelspacing=0.4, handlelength=1.5)
    for text in legend.get_texts():
        text.set_fontweight('normal')
    _auto_resize_legend(ax, legend)
    ax.grid(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    _set_bold_ticks(ax, 24)
    _set_spines(ax, 2.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def plot_combined_dca(y_true, models_data, output_path, title):
    fig, ax = plt.subplots(figsize=(16, 12))
    thresholds = np.linspace(0.01, 0.99, 200)
    y_arr, n = np.asarray(y_true), len(y_true)
    er = np.mean(y_arr)
    ax.plot(thresholds, [(er-(1-er)*(t/(1-t))) if t < 1 else 0 for t in thresholds], 'k--', lw=2, label='Treat All')
    ax.plot(thresholds, np.zeros_like(thresholds), 'k-', lw=2, label='Treat None')
    for name, y_prob, color in models_data:
        net_benefit = []
        for thresh in thresholds:
            pred_pos = (y_prob >= thresh).astype(int)
            tp = np.sum((pred_pos == 1) & (y_arr == 1))
            fp = np.sum((pred_pos == 1) & (y_arr == 0))
            nb = (tp/n) - (fp/n)*(thresh/(1-thresh)) if thresh < 1 else 0
            net_benefit.append(nb)
        ax.plot(thresholds, net_benefit, lw=2.5, color=color, label=name)
    ax.set_xlabel('Threshold Probability', fontsize=27, fontweight='bold')
    ax.set_ylabel('Net Benefit', fontsize=27, fontweight='bold')
    ax.set_title(title, fontsize=33, fontweight='bold', pad=15)
    _set_bold_ticks(ax, 24)
    legend = ax.legend(loc='upper right', fontsize=28, frameon=False,
                       ncol=1, labelspacing=0.4, handlelength=1.5)
    for text in legend.get_texts():
        text.set_fontweight('normal')
    _auto_resize_legend(ax, legend)
    ax.set_ylim([-0.1, 0.5])
    _set_spines(ax, 2.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def plot_confusion_matrix(cm, model_name, output_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'],
                ax=ax, linewidths=2, linecolor='black',
                annot_kws={"size": 24, "weight": "bold"})
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=30, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=28, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=28, fontweight='bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(24)
        label.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def plot_horizontal_bar(rank_data, output_path, title):
    fig, ax = plt.subplots(figsize=(16, max(10, len(rank_data) * 1.0)))
    names = [x[0] for x in rank_data]
    scores = [x[1] for x in rank_data]
    colors = [x[2] for x in rank_data]
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor='black', height=0.7)
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=27, fontweight='bold')
    ax.set_xlabel('Comprehensive Rank Score (lower is better)', fontsize=30, fontweight='bold')
    ax.set_title(title, fontsize=36, fontweight='bold', pad=15)
    ax.tick_params(axis='x', labelsize=27, width=2.5, length=8)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('black')
    # 数字注释已移除
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def validate_single_cohort(model_folder, ext_file, master_folder, cohort_idx):
    cohort_name = os.path.splitext(os.path.basename(ext_file))[0]
    output_folder = os.path.join(master_folder, f"{cohort_idx}_{cohort_name}")
    for sub in ['1_性能表格','2_统计检验','3_ROC曲线','4_PR曲线','5_DCA曲线',
                '6_校准曲线','7_混淆矩阵','8_汇总可视化']:
        os.makedirs(os.path.join(output_folder, sub), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"队列 [{cohort_idx}]: {cohort_name}")
    print(f"{'='*60}")

    feature_file = os.path.join(model_folder, 'feature_names.csv')
    scaler_file = os.path.join(model_folder, 'scaler.joblib')
    if not os.path.exists(feature_file): raise FileNotFoundError(f"缺少 {feature_file}")
    if not os.path.exists(scaler_file): raise FileNotFoundError(f"缺少 {scaler_file}")

    feature_names = pd.read_csv(feature_file)['feature_name'].tolist()
    scaler = joblib.load(scaler_file)
    print(f"  特征: {len(feature_names)} 个")

    models = []
    for i in range(1, 6):
        pattern = os.path.join(model_folder, f'{i}_*_model.joblib')
        files = glob.glob(pattern)
        if not files:
            print(f"  未找到第{i}个模型"); continue
        model_data = joblib.load(files[0])
        config_file = files[0].replace('_model.joblib', '_config.json')
        config = json.load(open(config_file, 'r', encoding='utf-8')) if os.path.exists(config_file) else {}
        models.append({
            'rank': i, 'name': config.get('model_name', f'Model_{i}'),
            'type': config.get('model_type', 'Base'),
            'model_data': model_data, 'config': config,
            'threshold': config.get('threshold', 0.5),
            'color': TOP5_COLORS[i-1]
        })
        print(f"  [{i}] {models[-1]['name']} (threshold={models[-1]['threshold']:.3f})")

    df_ext = smart_read_file(ext_file)
    print(f"  数据: {df_ext.shape}")

    has_target = TARGET_COL in df_ext.columns
    if has_target:
        y_ext = df_ext[TARGET_COL].values
        print(f"  阳性率: {np.mean(y_ext)*100:.1f}%")
    else:
        y_ext = None
        print(f"  目标变量缺失，仅输出概率")

    missing_feats = [f for f in feature_names if f not in df_ext.columns]
    if missing_feats: raise ValueError(f"外部数据缺少特征: {missing_feats}")
    X_ext = df_ext[feature_names].copy()
    missing_cnt = X_ext.isnull().sum().sum()
    if missing_cnt > 0:
        print(f"  警告: {missing_cnt} 个缺失值")
    X_ext_scaled = pd.DataFrame(scaler.transform(X_ext), columns=feature_names, index=X_ext.index)

    print("  预测...")
    results = []
    for m in models:
        md = m['model_data']
        if m['type'] == 'Base':
            y_prob = md['model'].predict_proba(X_ext_scaled)[:, 1]
        else:
            meta_dict = {}
            for bname, bmodel in md['base_models'].items():
                meta_dict[bname] = bmodel.predict_proba(X_ext_scaled)[:, 1]
            X_meta = pd.DataFrame(meta_dict)
            y_prob = md['meta_model'].predict_proba(X_meta)[:, 1]
        results.append({
            'rank': m['rank'], 'name': m['name'], 'type': m['type'],
            'y_prob': y_prob, 'threshold': m['threshold'], 'color': m['color'],
            'config': m['config']
        })

    if y_ext is None:
        prob_df = pd.DataFrame({r['name']: r['y_prob'] for r in results})
        prob_df.to_csv(os.path.join(output_folder, '1_性能表格', 'External_Predicted_Probabilities.csv'), index=False)
        print(f"  [INFO] 预测概率已保存")
        return None

    print("  性能评估...")
    perf_metrics = ['AUC','PR_AUC','Accuracy','F1_score','NPV','PPV','Sensitivity','Specificity','G_mean']
    ext_results = []
    for r in results:
        metrics, _ = compute_metrics_with_threshold(y_ext, r['y_prob'], threshold=r['threshold'])
        auc_str, _, _, _ = bootstrap_auc_ci(y_ext, r['y_prob'])
        pr_auc_str, _, _, _ = bootstrap_pr_auc_ci(y_ext, r['y_prob'])
        calib_metrics = compute_calibration_metrics(y_ext, r['y_prob'])
        ext_results.append({
            'Model': r['name'], 'Type': r['type'], 
            'AUC (95% CI)': auc_str, 'PR_AUC (95% CI)': pr_auc_str,
            **metrics, 'Brier_Score': calib_metrics['Brier'], 'Threshold': r['threshold']
        })
    df_perf = pd.DataFrame(ext_results)
    save_df(df_perf, os.path.join(output_folder, '1_性能表格', '表E1_外部验证5模型性能.csv'))

    print("  DeLong检验...")
    delong_records = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            z, p, auc1, auc2 = delong_test(y_ext, results[i]['y_prob'], results[j]['y_prob'])
            auc1_str = df_perf[df_perf['Model']==results[i]['name']]['AUC (95% CI)'].values[0]
            auc2_str = df_perf[df_perf['Model']==results[j]['name']]['AUC (95% CI)'].values[0]
            delong_records.append({
                'Method_1': results[i]['name'], 'Method_2': results[j]['name'],
                'AUC1 (95% CI)': auc1_str, 'AUC2 (95% CI)': auc2_str, 'Z': z, 'P': p
            })
    df_delong = pd.DataFrame(delong_records)
    df_delong = adjust_pvalues(df_delong, p_col='P', method='fdr_bh')
    save_df(df_delong, os.path.join(output_folder, '2_统计检验', '表E2_外部验证DeLong检验.csv'))

    print("  综合排名...")
    rank_df = df_perf.copy()
    rank_df, _ = comprehensive_ranking_with_ranks(rank_df, perf_metrics)
    save_df(rank_df[['Model','Type','AUC (95% CI)','AUC','AUC_rank','PR_AUC (95% CI)','PR_AUC','PR_AUC_rank',
                     'Accuracy','Accuracy_rank','F1_score','F1_score_rank',
                     'Sensitivity','Sensitivity_rank','Specificity','Specificity_rank',
                     'PPV','PPV_rank','NPV','NPV_rank','G_mean','G_mean_rank',
                     'Comprehensive_Rank']],
            os.path.join(output_folder, '1_性能表格', '表E4_外部验证5模型综合排名.csv'))

    print("  可视化...")
    roc_data, pr_data, cal_data, dca_data = [], [], [], []
    for r in results:
        safe = r['name'].replace(' ','_').replace('+','_')
        fpr, tpr, _ = roc_curve(y_ext, r['y_prob'])
        prec, rec, _ = precision_recall_curve(y_ext, r['y_prob'])

        plot_roc_clean(fpr, tpr, r['color'], r['name'], os.path.join(output_folder, '3_ROC曲线', f'{safe}_ROC.tiff'))
        plot_pr_clean(prec, rec, r['color'], r['name'], os.path.join(output_folder, '4_PR曲线', f'{safe}_PR.tiff'), y_true=y_ext, y_prob=r['y_prob'])
        plot_dca_standard(y_ext, r['y_prob'], r['color'], r['name'], os.path.join(output_folder, '5_DCA曲线', f'{safe}_DCA.tiff'))
        plot_calibration_fixed(y_ext, r['y_prob'], r['color'], r['name'], os.path.join(output_folder, '6_校准曲线', f'{safe}_Calibration.tiff'))

        cm = confusion_matrix(y_ext, (r['y_prob']>=r['threshold']).astype(int), labels=[0,1])
        plot_confusion_matrix(cm, r['name'], os.path.join(output_folder, '7_混淆矩阵', f'{safe}_CM.tiff'))

        roc_data.append((r['name'], fpr, tpr, r['color']))
        pr_data.append((r['name'], r['y_prob'], r['color']))
        cal_data.append((r['name'], r['y_prob'], r['color']))
        dca_data.append((r['name'], r['y_prob'], r['color']))

    plot_combined_roc(roc_data, os.path.join(output_folder, '8_汇总可视化', '图1_外部验证汇总ROC.tiff'), f'{cohort_name} - Top 5 Models')
    plot_combined_pr(pr_data, os.path.join(output_folder, '8_汇总可视化', '图2_外部验证汇总PR.tiff'), f'{cohort_name} - Top 5 Models', y_ext)
    plot_combined_calibration(y_ext, cal_data, os.path.join(output_folder, '8_汇总可视化', '图3_外部验证汇总校准.tiff'), f'{cohort_name} - Calibration')
    plot_combined_dca(y_ext, dca_data, os.path.join(output_folder, '8_汇总可视化', '图4_外部验证汇总DCA.tiff'), f'{cohort_name} - DCA')

    rank_plot_data = [(row['Model'], row['Comprehensive_Rank'], TOP5_COLORS[i]) 
                      for i, (_, row) in enumerate(rank_df.iterrows())]
    plot_horizontal_bar(rank_plot_data, os.path.join(output_folder, '8_汇总可视化', '图5_外部验证综合排名.tiff'), f'{cohort_name} - Comprehensive Rank')

    report_path = os.path.join(output_folder, 'Validation_Report.md')
    n_sample = len(y_ext)
    event_rate = np.mean(y_ext)
    best_row = rank_df.iloc[0]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 外部队列验证报告 — {cohort_name}\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**模型来源**: `{model_folder}`\n\n")
        f.write(f"**外部数据**: `{ext_file}`\n\n")
        f.write(f"**样本量**: {n_sample}\n\n")
        f.write(f"**阳性率**: {event_rate*100:.2f}%\n\n")
        f.write(f"**说明**\n\n")
        f.write(f"---\n\n")

        f.write(f"## 1. 模型性能摘要（表E1）\n\n")
        f.write(df_perf.to_markdown(index=False))
        f.write(f"\n\n")

        f.write(f"---\n\n")
        f.write(f"## 2. DeLong检验\n\n")
        f.write(df_delong.to_markdown(index=False))
        f.write(f"\n\n")

        f.write(f"---\n\n")
        f.write(f"## 3. 综合排名（表E4）\n\n")
        f.write(f"**最佳模型**: {best_row['Model']} (综合排名得分: {best_row['Comprehensive_Rank']:.3f})\n\n")
        f.write(rank_df[['Model','Type','AUC','PR_AUC','Comprehensive_Rank']].to_markdown(index=False))
        f.write(f"\n\n")

        f.write(f"---\n\n")
        f.write(f"## 4. 输出文件清单\n\n")
        f.write(f"```\n")
        f.write(f"{output_folder}/\n")
        f.write(f"├── 1_性能表格/\n")
        f.write(f"│   ├── 表E1_外部验证5模型性能.csv\n")
        f.write(f"│   └── 表E4_外部验证5模型综合排名.csv\n")
        f.write(f"├── 2_统计检验/\n")
        f.write(f"│   └── 表E2_外部验证DeLong检验.csv\n")
        f.write(f"├── 3_ROC曲线/\n")
        f.write(f"├── 4_PR曲线/\n")
        f.write(f"├── 5_DCA曲线/\n")
        f.write(f"├── 6_校准曲线/\n")
        f.write(f"├── 7_混淆矩阵/\n")
        f.write(f"└── 8_汇总可视化/\n")
        f.write(f"    ├── 图1_外部验证汇总ROC.tiff\n")
        f.write(f"    ├── 图2_外部验证汇总PR.tiff\n")
        f.write(f"    ├── 图3_外部验证汇总校准.tiff\n")
        f.write(f"    ├── 图4_外部验证汇总DCA.tiff\n")
        f.write(f"    └── 图5_外部验证综合排名.tiff\n")
        f.write(f"```\n\n")
        f.write(f"*报告自动生成*\n")

    print(f"  [INFO] 报告: {report_path}")

    return {
        'cohort_name': cohort_name,
        'n_sample': len(y_ext),
        'event_rate': np.mean(y_ext),
        'best_model': best_row['Model'],
        'best_auc': df_perf.loc[df_perf['Model']==best_row['Model'], 'AUC'].values[0],
        'df_perf': df_perf,
        'rank_df': rank_df
    }

def main():
    print(f"\n{'='*60}")
    print(f"代码包5：多外部队列验证")
    print(f"{'='*60}")

    model_folder = MODEL_FOLDER if MODEL_FOLDER and os.path.exists(MODEL_FOLDER) else auto_detect_model_folder()
    ext_files = auto_detect_external_files()

    today = datetime.now().strftime("%Y%m%d")
    master_folder = f"{today}_External_Validation"
    os.makedirs(master_folder, exist_ok=True)

    print(f"\n[INFO] 模型: {model_folder}")
    print(f"[INFO] 外部队列: {len(ext_files)} 个")
    print(f"[INFO] 输出: {master_folder}")
    print(f"{'='*60}")

    all_cohort_summaries = []
    for idx, ext_file in enumerate(ext_files, 1):
        summary = validate_single_cohort(model_folder, ext_file, master_folder, idx)
        if summary:
            all_cohort_summaries.append(summary)

    if all_cohort_summaries:
        report_path = os.path.join(master_folder, 'Summary_Report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 多外部队列验证汇总报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**模型来源**: `{model_folder}`\n\n")
            f.write(f"**验证队列数**: {len(all_cohort_summaries)}\n\n")
            f.write(f"**说明**\n\n")
            f.write(f"---\n\n")

            f.write(f"## 各队列基本信息\n\n")
            f.write(f"| 队列 | 样本量 | 阳性率 | 最佳模型 | AUC |\n")
            f.write(f"|------|--------|--------|----------|-----|\n")
            for s in all_cohort_summaries:
                f.write(f"| {s['cohort_name']} | {s['n_sample']} | {s['event_rate']*100:.2f}% | {s['best_model']} | {s['best_auc']:.3f} |\n")
            f.write(f"\n\n")

            for s in all_cohort_summaries:
                f.write(f"---\n\n")
                f.write(f"## {s['cohort_name']} 详细结果\n\n")
                f.write(f"### 性能摘要（表E1）\n\n")
                f.write(s['df_perf'][['Model','Type','AUC (95% CI)','AUC','PR_AUC (95% CI)','PR_AUC','Accuracy','F1_score','Sensitivity','Specificity','Brier_Score']].to_markdown(index=False))
                f.write(f"\n\n")
                f.write(f"### 综合排名（表E4）\n\n")
                f.write(s['rank_df'][['Model','Type','AUC','PR_AUC','Comprehensive_Rank']].to_markdown(index=False))
                f.write(f"\n\n")

            f.write(f"---\n\n")
            f.write(f"*报告自动生成*\n")

        print(f"\n{'='*60}")
        print(f"[DONE] 完成 {len(all_cohort_summaries)} 个外部队列验证")
        print(f"[DONE] 汇总报告: {report_path}")
        print(f"{'='*60}")

if __name__ == '__main__':
    main()