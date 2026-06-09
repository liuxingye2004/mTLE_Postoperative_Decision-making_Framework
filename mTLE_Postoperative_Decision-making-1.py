#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import chi2, norm, binom
from scipy import stats
from scipy.interpolate import PchipInterpolator
from itertools import combinations
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("statsmodels未安装，使用Bonferroni近似校正")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier

import joblib
import json

from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP未安装，跳过SHAP分析")

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

BASE_MODEL_COLORS = {
    'Logistic Regression': '#2E86AB', 'LDA': '#A23B72', 'Elastic Net': '#F18F01',
    'Gaussian NB': '#2D3047', 'Bernoulli NB': '#540D6E', 'Multinomial NB': '#8E44AD',
    'Gaussian Process': '#EE4266',
    'SVM': '#F4A261', 'KNN': '#264653',
    'ABPNN': '#FF1493', 'PLS-DA': '#00CED1', 'ExtraTrees': '#FF6347'
}
TOP5_COLORS = ['#D62828', '#F77F00', '#FCBF49', '#06A77D', '#118AB2']
MODEL_COLOR_MAP = {}

def get_stacking_color(base_names):
    colors = []
    for name in base_names:
        if name in BASE_MODEL_COLORS:
            colors.append(BASE_MODEL_COLORS[name])
    if not colors:
        return '#666666'
    rgb = [0, 0, 0]
    for c in colors:
        c_rgb = plt.cm.colors.to_rgb(c)
        rgb[0] += c_rgb[0]; rgb[1] += c_rgb[1]; rgb[2] += c_rgb[2]
    rgb = [x/len(colors) for x in rgb]
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def create_output_folder():
    today = datetime.now().strftime("%Y%m%d")
    version = 1
    while True:
        folder_name = f"{today}_kimi_v{version}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            subfolders = [
                '1_基础模型性能', '2_堆叠集成模型性能', '3_综合排名与比较',
                '4_ROC曲线', '4_PR曲线', '5_雷达图', '6_DCA曲线', '7_校准曲线',
                '8_SHAP分析', '9_交叉验证结果', '10_统计检验结果', '11_汇总可视化'
            ]
            for sub in subfolders:
                os.makedirs(os.path.join(folder_name, sub))
            print(f"创建输出文件夹：{folder_name}")
            return folder_name
        version += 1

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
        if threshold is None:
            fpr, tpr, thresholds = roc_curve(y_true_arr, y_pred_prob_arr)
            if len(thresholds) == 0:
                best_thresh = 0.5
            else:
                youden_index = tpr - fpr
                best_idx = np.argmax(youden_index)
                best_thresh = thresholds[best_idx]
            threshold = best_thresh
        y_pred = (y_pred_prob_arr >= threshold).astype(int)
        cm = confusion_matrix(y_true_arr, y_pred, labels=[0,1])
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = np.sum((y_true_arr == 0) & (y_pred == 0))
            fp = np.sum((y_true_arr == 0) & (y_pred == 1))
            fn = np.sum((y_true_arr == 1) & (y_pred == 0))
            tp = np.sum((y_true_arr == 1) & (y_pred == 1))
        sensitivity = tp / (tp+fn) if (tp+fn)>0 else 0
        specificity = tn / (tn+fp) if (tn+fp)>0 else 0
        ppv = tp / (tp+fp) if (tp+fp)>0 else 0
        npv = tn / (tn+fn) if (tn+fn)>0 else 0
        accuracy = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0
        f1 = 2*ppv*sensitivity/(ppv+sensitivity) if (ppv+sensitivity)>0 else 0
        g_mean = np.sqrt(sensitivity*specificity)
        return {
            'AUC': round(auc_val,3), 'Accuracy': round(accuracy,3),
            'F1_score': round(f1,3), 'NPV': round(npv,3), 'PPV': round(ppv,3),
            'Sensitivity': round(sensitivity,3), 'Specificity': round(specificity,3),
            'G_mean': round(g_mean,3), 'Threshold': round(threshold,3)
        }, float(threshold)
    except Exception as e:
        return {
            'AUC':0,'Accuracy':0,'F1_score':0,'NPV':0,'PPV':0,
            'Sensitivity':0,'Specificity':0,'G_mean':0,'Threshold':0.5
        }, 0.5

def brier_score(y_true, y_pred_prob):
    return round(np.mean((np.asarray(y_true) - np.asarray(y_pred_prob))**2), 3)

def _quantile_calibration_bins(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n = len(y_true)
    if n_bins is None:
        n_bins = min(10, max(5, n // 20))
    if n < n_bins * 2:
        n_bins = max(3, n // 2)
    order = np.argsort(y_prob)
    y_prob_sorted = y_prob[order]
    y_true_sorted = y_true[order]
    bin_size = n // n_bins
    prob_pred_list = []
    prob_true_list = []
    counts_list = []
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
        np.random.seed(42)
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred_prob)
        n = len(y_true_arr)
        theta_hat = roc_auc_score(y_true_arr, y_pred_arr)
        aucs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            if len(np.unique(y_true_arr[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y_true_arr[idx], y_pred_arr[idx]))
        if len(aucs) < 100:
            return f"{theta_hat:.3f} (NA)", theta_hat, np.nan, np.nan
        aucs = np.array(aucs)
        z0 = norm.ppf(np.mean(aucs < theta_hat))
        alpha1 = norm.cdf(z0 + (z0 + norm.ppf(alpha/2)))
        alpha2 = norm.cdf(z0 + (z0 + norm.ppf(1-alpha/2)))
        lower = np.percentile(aucs, alpha1 * 100)
        upper = np.percentile(aucs, alpha2 * 100)
        lower = max(0.0, min(1.0, lower))
        upper = max(0.0, min(1.0, upper))
        return f"{theta_hat:.3f} ({lower:.3f}-{upper:.3f})", theta_hat, lower, upper
    except Exception:
        try:
            theta_hat = roc_auc_score(y_true, y_pred_prob)
            return f"{theta_hat:.3f} (Error)", theta_hat, 0.5, 0.5
        except:
            return "0.500 (Error)", 0.5, 0.5, 0.5

def delong_test(y_true, y_pred_prob1, y_pred_prob2):
    y_true = np.asarray(y_true)
    scores1 = np.asarray(y_pred_prob1)
    scores2 = np.asarray(y_pred_prob2)
    auc1 = roc_auc_score(y_true, scores1)
    auc2 = roc_auc_score(y_true, scores2)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return 0.0, 1.0, round(auc1,3), round(auc2,3)
    V10_1 = np.zeros(n_pos); V10_2 = np.zeros(n_pos)
    for i, idx in enumerate(pos_idx):
        s1 = scores1[idx]; s2 = scores2[idx]
        V10_1[i] = np.mean(scores1[neg_idx] < s1) + 0.5 * np.mean(scores1[neg_idx] == s1)
        V10_2[i] = np.mean(scores2[neg_idx] < s2) + 0.5 * np.mean(scores2[neg_idx] == s2)
    V01_1 = np.zeros(n_neg); V01_2 = np.zeros(n_neg)
    for i, idx in enumerate(neg_idx):
        s1 = scores1[idx]; s2 = scores2[idx]
        V01_1[i] = np.mean(scores1[pos_idx] > s1) + 0.5 * np.mean(scores1[pos_idx] == s1)
        V01_2[i] = np.mean(scores2[pos_idx] > s2) + 0.5 * np.mean(scores2[pos_idx] == s2)
    S01_1 = np.var(V01_1, ddof=1) if n_neg > 1 else 0
    S10_1 = np.var(V10_1, ddof=1) if n_pos > 1 else 0
    S01_2 = np.var(V01_2, ddof=1) if n_neg > 1 else 0
    S10_2 = np.var(V10_2, ddof=1) if n_pos > 1 else 0
    S01_12 = np.cov(V01_1, V01_2, ddof=1)[0,1] if n_neg > 1 else 0
    S10_12 = np.cov(V10_1, V10_2, ddof=1)[0,1] if n_pos > 1 else 0
    var_auc1 = S01_1 / n_neg + S10_1 / n_pos
    var_auc2 = S01_2 / n_neg + S10_2 / n_pos
    cov_auc = S01_12 / n_neg + S10_12 / n_pos
    var_diff = var_auc1 + var_auc2 - 2 * cov_auc
    if var_diff <= 0:
        var_diff = 1e-10
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = 2 * (1 - norm.cdf(abs(z)))
    return round(z, 3), round(p, 3), round(auc1, 3), round(auc2, 3)

def adjust_pvalues(df, p_col='P', method='fdr_bh'):
    if p_col not in df.columns:
        return df
    pvals = df[p_col].astype(float).values
    n_tests = len(pvals)
    if STATSMODELS_AVAILABLE:
        try:
            reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method=method, is_sorted=False, returnsorted=False)
            adj_col = f'{p_col}_FDR'
            df[adj_col] = np.round(p_adj, 3)
            df['Significant_FDR'] = reject
        except Exception:
            adj_col = f'{p_col}_Bonferroni'
            df[adj_col] = np.round(np.minimum(pvals * n_tests, 1.0), 3)
            df['Significant_Bonf'] = df[adj_col] < 0.05
    else:
        adj_col = f'{p_col}_Bonferroni'
        df[adj_col] = np.round(np.minimum(pvals * n_tests, 1.0), 3)
        df['Significant_Bonf'] = df[adj_col] < 0.05

    for sig_col in ['Significant_FDR', 'Significant_Bonf']:
        if sig_col in df.columns:
            df[sig_col] = df[sig_col].apply(lambda x: '**TRUE**' if pd.notna(x) and bool(x) else 'FALSE')

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

def generate_stacking_features_cv(base_models, X_train, y_train, X_test, cv=5, n_repeats=2):
    rskf = RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats, random_state=42)
    train_meta = pd.DataFrame(index=X_train.index)
    for name, model in base_models.items():
        print(f"    生成 {name} 的OOF元特征 (Repeated CV {cv}x{n_repeats})...")
        oof_preds = np.zeros(len(y_train))
        oof_counts = np.zeros(len(y_train))
        for train_idx, val_idx in rskf.split(X_train, y_train):
            X_tr_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr_fold = y_train.iloc[train_idx]
            try:
                model_clone = clone(model)
                model_clone.fit(X_tr_fold, y_tr_fold)
                probs = model_clone.predict_proba(X_val_fold)[:, 1]
                oof_preds[val_idx] += probs
                oof_counts[val_idx] += 1
            except Exception as e:
                print(f"      警告: {name} 折叠训练失败: {str(e)[:50]}")
        oof_preds = np.divide(oof_preds, oof_counts, out=np.zeros_like(oof_preds), where=oof_counts!=0)
        oof_preds[oof_counts == 0] = 0.5
        train_meta[name] = oof_preds
    return train_meta

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
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

def _set_bold_ticks(ax, labelsize=24):
    ax.tick_params(axis='both', labelsize=labelsize, width=2.5, length=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

def _set_spines(ax, lw=2.5):
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
        spine.set_color('black')

def _auto_resize_legend(ax, legend, min_fontsize=10):
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
    ax.plot(fpr, tpr, color=color, lw=3, label=f'{model_name} (AUC = {auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Reference')
    ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=24, fontweight='bold')
    ax.set_title(f'ROC Curve - {model_name}', fontsize=30, fontweight='bold', pad=15)
    _set_bold_ticks(ax, 24)
    ax.grid(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    _set_spines(ax, 2.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def plot_pr_clean(precision, recall, color, model_name, output_path, y_true=None, y_prob=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    if y_prob is not None and y_true is not None:
        ap_val = average_precision_score(y_true, y_prob)
    else:
        ap_val = np.sum((recall[:-1] - recall[1:]) * precision[:-1]) if len(precision) > 1 else 0
    ax.plot(recall, precision, color=color, lw=3, label=f'{model_name} (AP = {ap_val:.3f})')
    if y_true is not None:
        baseline = np.mean(y_true)
        ax.axhline(y=baseline, color='gray', linestyle='--', lw=2, alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Precision (PPV)', fontsize=24, fontweight='bold')
    ax.set_title(f'PR Curve - {model_name}', fontsize=30, fontweight='bold', pad=15)
    _set_bold_ticks(ax, 24)
    ax.grid(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    _set_spines(ax, 2.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def plot_radar_clean(angles, metrics_norm, metric_names, color, model_name, output_path):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, metrics_norm, color=color, linewidth=3)
    ax.fill(angles, metrics_norm, alpha=0.25, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.yaxis.grid(True, linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
    ax.xaxis.grid(True, linestyle='-', linewidth=1.0, color='gray', alpha=0.5)
    ax.spines['polar'].set_linewidth(1.0)
    ax.spines['polar'].set_color('black')
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
        ax.set_xlabel('Threshold Probability', fontsize=24, fontweight='bold')
        ax.set_ylabel('Net Benefit', fontsize=24, fontweight='bold')
        ax.set_title(f'Decision Curve Analysis - {model_name}', fontsize=30, fontweight='bold', pad=15)
        _set_bold_ticks(ax, 24)
        ax.grid(False)
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.1, 0.25])
        _set_spines(ax, 2.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
        plt.close()
    except Exception as e:
        print(f"DCA曲线绘制失败: {str(e)[:50]}")

def plot_calibration_curve(y_true, y_prob, color, model_name, output_path, n_bins=None):
    try:
        if n_bins is None:
            n_bins = min(10, max(5, len(y_true) // 20))
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
        ax.set_xlabel('Mean Predicted Probability', fontsize=24, fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontsize=24, fontweight='bold')
        ax.set_title(f'Calibration Curve - {model_name}', fontsize=30, fontweight='bold', pad=15)
        _set_bold_ticks(ax, 24)
        ax.grid(False)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal', adjustable='box')
        _set_spines(ax, 2.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
        plt.close()
    except Exception as e:
        print(f"校准曲线绘制失败: {str(e)[:50]}")

def plot_combined_roc(models_data, output_path, title):
    fig, ax = plt.subplots(figsize=(16, 12))
    for name, fpr, tpr, color in models_data:
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} ({auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Reference')
    ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=27, fontweight='bold')
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=27, fontweight='bold')
    ax.set_title(title, fontsize=33, fontweight='bold', pad=15)
    _set_bold_ticks(ax, 24)
    ax.grid(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    _set_spines(ax, 2.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def plot_combined_pr(models_data, output_path, title, y_true):
    fig, ax = plt.subplots(figsize=(16, 12))
    baseline = np.mean(y_true)
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=2, alpha=0.5, label=f'Baseline ({baseline:.3f})')
    for name, y_prob, color in models_data:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color=color, lw=2.5, label=f'{name} (AP={ap_val:.3f})')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=27, fontweight='bold')
    ax.set_ylabel('Precision (PPV)', fontsize=27, fontweight='bold')
    ax.set_title(title, fontsize=33, fontweight='bold', pad=15)
    _set_bold_ticks(ax, 24)
    ax.grid(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    _set_spines(ax, 2.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def plot_combined_calibration_curve(y_true, models_data, output_path, title, n_bins=None):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7, label='Ideal')
    if n_bins is None:
        n_bins = min(10, max(5, len(y_true) // 20))
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
    ax.set_xlabel('Mean Predicted Probability', fontsize=27, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=27, fontweight='bold')
    ax.set_title(title, fontsize=33, fontweight='bold', pad=15)
    ax.grid(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal', adjustable='box')
    _set_bold_ticks(ax, 24)
    _set_spines(ax, 2.5)
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
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

class SHAPAnalyzer:
    def __init__(self, X_val, feature_names, output_folder, model_color=None, global_xlim=None):
        if isinstance(X_val, pd.DataFrame):
            self.X_val = X_val.values.copy()
            self.feature_names = X_val.columns.tolist()
        else:
            self.X_val = np.array(X_val, copy=True)
            self.feature_names = feature_names if feature_names else [f'Feature_{i}' for i in range(self.X_val.shape[1])]
        self.output_folder = output_folder
        self.model_color = model_color if model_color else '#2E86AB'
        self.global_xlim = global_xlim
        os.makedirs(self.output_folder, exist_ok=True)

    def analyze_model(self, model_name, model):
        print(f"  正在对 {model_name} 进行SHAP分析...")
        try:
            shap_values = self._compute_shap_values(model)
            if shap_values is None:
                return False
            self._plot_shap_summary_bar(model_name, shap_values)
            self._plot_shap_beeswarm(model_name, shap_values)
            print(f"    {model_name} SHAP分析完成")
            return True
        except Exception as e:
            print(f"    {model_name} SHAP分析失败: {str(e)[:100]}")
            return False

    def _compute_shap_values(self, model):
        try:
            model_type = type(model).__name__
            if any(tree in model_type for tree in ['RandomForest','ExtraTrees','DecisionTree',
                                                    'GradientBoosting','XGBClassifier',
                                                    'CatBoostClassifier','LGBM']):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(self.X_val)
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values = shap_values[1]
            elif any(lin in model_type for lin in ['LogisticRegression','LinearDiscriminant','SGDClassifier']):
                explainer = shap.LinearExplainer(model, self.X_val)
                shap_values = explainer.shap_values(self.X_val)
            else:
                def model_predict(X):
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        return proba[:,1] if proba.ndim > 1 else proba
                    return model.predict(X)
                rng = np.random.RandomState(42)
                bg_idx = rng.choice(self.X_val.shape[0], min(10, self.X_val.shape[0]), replace=False)
                background = self.X_val[bg_idx]
                explainer = shap.KernelExplainer(model_predict, background)
                shap_values = explainer.shap_values(self.X_val, nsamples=100)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            return np.array(shap_values)
        except Exception as e:
            print(f"    SHAP计算失败: {str(e)[:80]}")
            return None

    def _plot_shap_summary_bar(self, model_name, shap_values):
        try:
            fig, ax = plt.subplots(figsize=(16, 14))
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            indices = np.argsort(mean_abs_shap)[::-1]
            features = [self.feature_names[i] for i in indices]
            values = mean_abs_shap[indices]
            colors = plt.cm.Reds(np.linspace(0.9, 0.3, len(values)))
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, values, color=colors, edgecolor='black', height=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=27, fontweight='bold')
            ax.invert_yaxis()
            ax.set_xlabel('mean(|SHAP value|)', fontsize=36, fontweight='bold')
            ax.set_title(f'SHAP Feature Importance - {model_name}', fontsize=30, fontweight='bold', pad=15)
            ax.tick_params(axis='x', labelsize=36, width=2.5, length=8)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_linewidth(3)
                spine.set_color('black')
            plt.tight_layout()
            safe_name = model_name.replace('+', '_').replace(' ', '_')
            output_path = os.path.join(self.output_folder, f'{safe_name}_SHAP_Bar.tiff')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
            plt.close()
            print(f"    [INFO] 柱状图已保存: {output_path}")
        except Exception as e:
            print(f"    SHAP Bar失败: {str(e)[:100]}")

    def _plot_shap_beeswarm(self, model_name, shap_values):
        try:
            fig, ax = plt.subplots(figsize=(20, 16))
            shap.summary_plot(shap_values, self.X_val, feature_names=self.feature_names,
                              plot_type="dot", show=False, color=plt.cm.coolwarm,
                              max_display=len(self.feature_names))
            ax = plt.gca()
            ax.tick_params(axis='y', labelsize=9, width=2.5, length=8)
            ax.tick_params(axis='x', labelsize=36, width=2.5, length=8)
            for label in ax.get_yticklabels():
                label.set_fontweight('normal')
            for label in ax.get_xticklabels():
                label.set_fontweight('normal')
            ax.set_xlabel('')
            ax.set_title('')
            if self.global_xlim:
                ax.set_xlim(-self.global_xlim, self.global_xlim)
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
            plt.tight_layout()
            safe_name = model_name.replace('+', '_').replace(' ', '_')
            output_path = os.path.join(self.output_folder, f'{safe_name}_SHAP_Beeswarm.tiff')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
            plt.close()
            print(f"    [INFO] 蜂状图已保存: {output_path}")
        except Exception as e:
            print(f"    SHAP Beeswarm失败: {str(e)[:100]}")

def main():
    output_folder = create_output_folder()

    search_patterns_train = [
        './output_feature_selection/*train*.csv',
        './output_feature_selection/*selected*train*.csv',
        './output_imputation/*train*.csv',
        './output_imputation/*imputed*train*.csv',
        './*/*train*.csv',
        './*train*.csv'
    ]
    search_patterns_test = [
        './output_feature_selection/*test*.csv',
        './output_feature_selection/*selected*test*.csv',
        './output_imputation/*test*.csv',
        './output_imputation/*imputed*test*.csv',
        './*/*test*.csv',
        './*test*.csv'
    ]

    train_candidates = []
    for pattern in search_patterns_train:
        train_candidates.extend(glob.glob(pattern))
    train_candidates = list(set(train_candidates))
    train_candidates = [p for p in train_candidates if 'external' not in p.lower()]
    if train_candidates:
        train_candidates.sort(key=os.path.getmtime, reverse=True)

    test_candidates = []
    for pattern in search_patterns_test:
        test_candidates.extend(glob.glob(pattern))
    test_candidates = list(set(test_candidates))
    test_candidates = [p for p in test_candidates if 'external' not in p.lower()]
    if test_candidates:
        test_candidates.sort(key=os.path.getmtime, reverse=True)

    alt_train_paths = [
        './output_feature_selection/selected_train.csv',
        './output_feature_selection/train_selected.csv',
        './output_feature_selection/selected_train_mice.csv',
        './output_feature_selection/train_selected_mice.csv',
        './output_imputation/best_imputed_train.csv',
        './output_imputation/best_imputed_train_mice.csv',
        './output_imputation/imputed_train_mice.csv',
        './output_imputation/train_imputed.csv',
    ]
    alt_test_paths = [
        './output_feature_selection/selected_test.csv',
        './output_feature_selection/test_selected.csv',
        './output_feature_selection/selected_test_mice.csv',
        './output_feature_selection/test_selected_mice.csv',
        './output_imputation/best_imputed_test.csv',
        './output_imputation/best_imputed_test_mice.csv',
        './output_imputation/imputed_test_mice.csv',
        './output_imputation/test_imputed.csv',
    ]

    train_path = None
    for p in train_candidates + alt_train_paths:
        if os.path.exists(p):
            train_path = p
            print(f"[INFO] 找到训练集: {p}")
            break
    test_path = None
    for p in test_candidates + alt_test_paths:
        if os.path.exists(p):
            test_path = p
            print(f"[INFO] 找到测试集: {p}")
            break

    if train_path is None:
        all_csv = glob.glob('./*/*.csv') + glob.glob('./*.csv')
        print("\n[ERR] 未找到训练集文件。当前目录下所有CSV文件：")
        for f in sorted(all_csv)[:30]:
            print(f"  - {f}")
        raise FileNotFoundError("未找到训练集文件。请确认代码包2已运行。")
    if test_path is None:
        raise FileNotFoundError("未找到测试集文件。请确认代码包2已运行。")

    train_data = pd.read_csv(train_path)
    test_data  = pd.read_csv(test_path)

    TARGET_COL = 'FUFA'
    if TARGET_COL not in train_data.columns:
        raise ValueError(f"数据中未找到'{TARGET_COL}'列")

    y_train = train_data[TARGET_COL]
    y_test  = test_data[TARGET_COL]
    X_train = train_data.drop(columns=[TARGET_COL])
    X_test  = test_data.drop(columns=[TARGET_COL])
    feature_names = X_train.columns.tolist()

    print(f"[INFO] 训练集: {len(y_train)} 例, 测试集: {len(y_test)} 例")
    print(f"[INFO] 核心特征数: {len(feature_names)}")
    print(f"[INFO] 特征列表: {feature_names}")

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test  = pd.DataFrame(scaler.transform(X_test),  columns=feature_names, index=X_test.index)

    print("\n[INFO] 初始化12个基础模型...")

    model_definitions = [
        ('Logistic Regression', 
         LogisticRegression(max_iter=10000, random_state=42, class_weight='balanced'), 
         {'C': [0.01, 0.1, 1.0, 10.0], 
          'solver': ['liblinear']}),

        ('LDA', 
         LinearDiscriminantAnalysis(), 
         {'solver': ['svd']}),

        ('Elastic Net', 
         LogisticRegression(penalty='elasticnet', solver='saga', max_iter=20000, 
                            random_state=42, class_weight='balanced'), 
         {'C': [0.01, 0.1, 1.0, 10.0], 
          'l1_ratio': [0.1, 0.5, 0.9]}),

        ('Gaussian NB', 
         GaussianNB(), 
         {'var_smoothing': [1e-11, 1e-10, 1e-9]}),

        ('Bernoulli NB', 
         BernoulliNBWrapper(alpha=1.0), 
         {'alpha': [0.1, 0.5, 1.0, 2.0]}),

        ('Multinomial NB', 
         MultinomialNBWrapper(alpha=1.0), 
         {'alpha': [0.1, 0.5, 1.0, 2.0]}),

        ('Gaussian Process', 
         GaussianProcessClassifier(random_state=42, max_iter_predict=100, 
                                    optimizer='fmin_l_bfgs_b',
                                    kernel=1.0 * RBF(length_scale_bounds=(0.1, 10.0))), 
         {'n_restarts_optimizer': [2, 5]}),

        ('SVM', 
         SVC(probability=True, random_state=42, class_weight='balanced'), 
         {'C': [0.1, 1.0, 10.0], 
          'gamma': ['scale', 'auto', 0.01, 0.1], 
          'kernel': ['rbf', 'linear']}),

        ('KNN', 
         KNeighborsClassifier(), 
         {'n_neighbors': [5, 7, 9, 11, 15], 
          'weights': ['uniform', 'distance'], 
          'p': [1, 2]}),

        ('ABPNN', 
         MLPClassifier(random_state=RANDOM_STATE, early_stopping=True, 
                       validation_fraction=0.2), 
         {'hidden_layer_sizes': [(25,), (50,)], 
          'activation': ['relu', 'tanh'],
          'alpha': [0.0001, 0.001, 0.01], 
          'learning_rate_init': [0.001, 0.01], 
          'max_iter': [2000]}),

        ('PLS-DA', 
         PLSDAClassifier(random_state=RANDOM_STATE), 
         {'n_components': [1, 2, 3, 4]}),

        ('ExtraTrees', 
         ExtraTreesClassifier(random_state=RANDOM_STATE,
                              class_weight='balanced'), 
         {'n_estimators': [100, 200], 
          'max_depth': [None, 10, 15],
          'min_samples_leaf': [2, 5, 10], 
          'max_features': ['sqrt', 'log2']})
    ]
    print(f"共初始化 {len(model_definitions)} 个基础模型")

    print("\n[INFO] 开始10-fold GridSearchCV交叉验证...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    base_results_train = []
    base_results_final = []
    base_models_trained = {}
    cv_predictions = {}
    train_cv_details = {}
    model_thresholds = {}

    for model_info in model_definitions:
        name = model_info[0]
        print(f"  训练: {name}")
        try:
            model = model_info[1]
            param_grid = model_info[2]

            oof_predictions = np.zeros(len(y_train))
            fold_performances = []
            best_params = {}

            if len(param_grid) > 0:
                try:
                    grid = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', 
                                        n_jobs=1, refit=False)
                    grid.fit(X_train, y_train)
                    best_params = grid.best_params_
                    print(f"    最优参数: {best_params}")
                except Exception as e:
                    print(f"    警告: GridSearchCV失败: {str(e)[:50]}")

            final_model = clone(model)
            if best_params:
                final_model.set_params(**best_params)
            final_model.fit(X_train, y_train)
            base_models_trained[name] = final_model

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                fold_model = clone(final_model)
                fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])

                if hasattr(fold_model, 'predict_proba'):
                    y_val_prob = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
                else:
                    y_val_prob = fold_model.predict(X_train.iloc[val_idx])

                oof_predictions[val_idx] = y_val_prob

                fold_metrics, _ = compute_metrics_with_threshold(y_train.iloc[val_idx], y_val_prob)
                fold_metrics['Fold'] = fold_idx + 1
                fold_performances.append(fold_metrics)

            cv_predictions[name] = oof_predictions
            train_cv_details[name] = fold_performances

            oof_metrics, oof_thresh = compute_metrics_with_threshold(y_train, oof_predictions)
            model_thresholds[name] = oof_thresh
            oof_auc_str, _, _, _ = bootstrap_auc_ci(y_train, oof_predictions)

            if hasattr(final_model, 'predict_proba'):
                y_final_prob = final_model.predict_proba(X_test)[:, 1]
            else:
                y_final_prob = final_model.predict(X_test)
            final_metrics, _ = compute_metrics_with_threshold(y_test, y_final_prob, threshold=oof_thresh)
            final_auc_str, _, _, _ = bootstrap_auc_ci(y_test, y_final_prob)

            calib_metrics = compute_calibration_metrics(y_test, y_final_prob)

            base_results_train.append({
                'Model': name, 'AUC (95% CI)': oof_auc_str, **oof_metrics, 'Threshold': oof_thresh
            })
            base_results_final.append({
                'Model': name, 'AUC (95% CI)': final_auc_str, **final_metrics,
                'Brier_Score': calib_metrics['Brier'], 'Threshold': oof_thresh
            })
        except Exception as e:
            print(f"    模型 {name} 训练失败: {str(e)[:80]}")
            continue

    if not base_results_train:
        print("错误：没有模型成功训练")
        return

    df_base_train = pd.DataFrame(base_results_train)
    df_base_final = pd.DataFrame(base_results_final)
    print(f"\n成功训练 {len(base_models_trained)} 个基础模型")

    performance_metrics = ['AUC','Accuracy','F1_score','NPV','PPV','Sensitivity','Specificity','G_mean']

    save_path_1 = os.path.join(output_folder, '1_基础模型性能', '表1_训练集CV_12基础模型性能.csv')
    col_order_1 = ['Model', 'AUC (95% CI)'] + performance_metrics + ['Threshold']
    save_df(df_base_train[col_order_1], save_path_1)

    save_path_2 = os.path.join(output_folder, '1_基础模型性能', '表2_最终测试集_12基础模型性能.csv')
    col_order_2 = ['Model', 'AUC (95% CI)'] + performance_metrics + [
        'Brier_Score', 'Threshold'
    ]
    save_df(df_base_final[col_order_2], save_path_2)

    print("\n生成表3：十折交叉验证结果汇总...")
    cv_summary = []
    for name in base_models_trained.keys():
        if name in train_cv_details:
            folds = train_cv_details[name]
            auc_values = [f['AUC'] for f in folds]
            mean_auc = np.mean(auc_values)
            ci_str, _, _, _ = bootstrap_auc_ci(y_train, cv_predictions[name], n_bootstrap=500)
            cv_summary.append({
                'Model': name, 'AUC (95% CI)': ci_str, 'AUC_mean': f"{mean_auc:.3f}",
                'Accuracy': f"{np.mean([f['Accuracy'] for f in folds]):.3f}",
                'F1_score': f"{np.mean([f['F1_score'] for f in folds]):.3f}",
                'NPV': f"{np.mean([f['NPV'] for f in folds]):.3f}",
                'PPV': f"{np.mean([f['PPV'] for f in folds]):.3f}",
                'Sensitivity': f"{np.mean([f['Sensitivity'] for f in folds]):.3f}",
                'Specificity': f"{np.mean([f['Specificity'] for f in folds]):.3f}",
                'G_mean': f"{np.mean([f['G_mean'] for f in folds]):.3f}"
            })
    save_path_3 = os.path.join(output_folder, '9_交叉验证结果', '表3_十折交叉验证结果汇总.csv')
    save_df(pd.DataFrame(cv_summary), save_path_3)

    print("\n进行基础模型间 DeLong 检验...")
    model_names = list(base_models_trained.keys())
    delong_base_records = []
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            name1 = model_names[i]
            name2 = model_names[j]
            prob1 = base_models_trained[name1].predict_proba(X_test)[:,1]
            prob2 = base_models_trained[name2].predict_proba(X_test)[:,1]
            z, p, auc1, auc2 = delong_test(y_test, prob1, prob2)
            auc1_str = df_base_final[df_base_final['Model']==name1]['AUC (95% CI)'].values[0]
            auc2_str = df_base_final[df_base_final['Model']==name2]['AUC (95% CI)'].values[0]
            delong_base_records.append({
                'Method_1': name1, 'Method_2': name2,
                'AUC1 (95% CI)': auc1_str, 'AUC2 (95% CI)': auc2_str,
                'Z': z, 'P': p
            })
    df_delong_base = pd.DataFrame(delong_base_records)
    df_delong_base = adjust_pvalues(df_delong_base, p_col='P', method='fdr_bh')
    save_path_s12 = os.path.join(output_folder, '10_统计检验结果', '表S12_基础模型DeLong检验.csv')
    save_df(df_delong_base, save_path_s12)
    print(f"[INFO] 已保存表S12")

    print("\n[INFO] 计算12基础模型综合排名...")

    base_rank_df = df_base_final.copy()
    base_rank_df, rank_cols_base = comprehensive_ranking_with_ranks(base_rank_df, performance_metrics)

    save_path_8a = os.path.join(output_folder, '3_综合排名与比较', '表8a_12基础模型综合排名.csv')
    rank_cols_all = ['Model', 'AUC (95% CI)', 'AUC', 'AUC_rank',
                     'Accuracy', 'Accuracy_rank',
                     'F1_score', 'F1_score_rank', 'G_mean', 'G_mean_rank',
                     'NPV', 'NPV_rank', 'PPV', 'PPV_rank', 'Sensitivity', 'Sensitivity_rank',
                     'Specificity', 'Specificity_rank', 'Comprehensive_Rank']
    rank_cols_all = [c for c in rank_cols_all if c in base_rank_df.columns]
    save_df(base_rank_df[rank_cols_all], save_path_8a)
    print("已保存表8a")

    print("\n[INFO] 选取Top5基础模型用于Stacking...")
    non_overfit_df = base_rank_df.copy()
    top5_base = non_overfit_df.head(5)['Model'].tolist()
    for i, name in enumerate(top5_base):
        row = non_overfit_df[non_overfit_df['Model'] == name].iloc[0]
        MODEL_COLOR_MAP[name] = TOP5_COLORS[i]
        print(f"    [{i+1}/5] {name}: Comprehensive_Rank={row['Comprehensive_Rank']:.3f}, "
              f"AUC={row['AUC']:.3f}")

    print("\n构建堆叠集成模型...")

    meta_train = generate_stacking_features_cv(
        base_models_trained, X_train, y_train, X_test, cv=5, n_repeats=2)

    meta_test_final = pd.DataFrame(index=X_test.index)
    for name, model in base_models_trained.items():
        try:
            meta_test_final[name] = model.predict_proba(X_test)[:, 1]
        except Exception:
            meta_test_final[name] = np.zeros(len(X_test)) + 0.5

    top5_models_dict = {name: base_models_trained[name] for name in top5_base}

    all_combinations = []
    for r in range(2, min(6, len(top5_base)+1)):
        all_combinations.extend(list(combinations(top5_base, r)))

    print(f"  基于Top5生成 {len(all_combinations)} 种堆叠组合")

    stacking_results = []
    stacking_models = {}
    for idx, comb in enumerate(all_combinations):
        comb_name = '+'.join(comb)
        print(f"  训练堆叠模型: {comb_name}")
        try:
            X_meta_train = meta_train[list(comb)]
            X_meta_test_final = meta_test_final[list(comb)]

            meta_model_raw = LogisticRegression(max_iter=2000, random_state=42, C=1.0, penalty='l2', solver='lbfgs')
            meta_model = meta_model_raw
            meta_model.fit(X_meta_train, y_train)

            stack_train_prob = meta_model.predict_proba(X_meta_train)[:, 1]
            _, stack_thresh = compute_metrics_with_threshold(y_train, stack_train_prob)
            model_thresholds[comb_name] = stack_thresh

            y_final_stack_prob = meta_model.predict_proba(X_meta_test_final)[:, 1]

            stacking_models[comb_name] = {
                'meta_model': meta_model,
                'base_models': {name: top5_models_dict[name] for name in comb},
                'combination': comb,
                'y_prob_test': y_final_stack_prob
            }

            stack_final_metrics, _ = compute_metrics_with_threshold(y_test, y_final_stack_prob, threshold=stack_thresh)
            stack_final_auc_str, _, _, _ = bootstrap_auc_ci(y_test, y_final_stack_prob)

            calib_metrics = compute_calibration_metrics(y_test, y_final_stack_prob)

            stacking_results.append({
                'Combination': comb_name, 'AUC (95% CI)': stack_final_auc_str,
                **stack_final_metrics,
                'Brier_Score': calib_metrics['Brier']
            })
        except Exception as e:
            print(f"    堆叠模型 {comb_name} 失败: {str(e)[:80]}")
            continue

    if not stacking_results:
        print("警告：没有堆叠模型成功训练")
        return

    df_stack = pd.DataFrame(stacking_results).sort_values('AUC', ascending=False)

    save_path_6 = os.path.join(output_folder, '2_堆叠集成模型性能', '表6_堆叠集成模型性能.csv')
    stack_perf_cols = ['Combination', 'AUC (95% CI)', 'AUC', 'Accuracy', 'F1_score', 'NPV', 'PPV',
                       'Sensitivity', 'Specificity', 'G_mean', 'Brier_Score']
    stack_perf_cols = [c for c in stack_perf_cols if c in df_stack.columns]
    save_df(df_stack[stack_perf_cols], save_path_6)
    print("已保存表6")

    print("\n进行堆叠模型间 DeLong 检验...")
    stack_names = list(stacking_models.keys())
    delong_stack_records = []
    for i in range(len(stack_names)):
        for j in range(i+1, len(stack_names)):
            name1 = stack_names[i]
            name2 = stack_names[j]
            prob1 = stacking_models[name1]['y_prob_test']
            prob2 = stacking_models[name2]['y_prob_test']
            z, p, auc1, auc2 = delong_test(y_test, prob1, prob2)
            auc1_str = df_stack[df_stack['Combination']==name1]['AUC (95% CI)'].values[0]
            auc2_str = df_stack[df_stack['Combination']==name2]['AUC (95% CI)'].values[0]
            delong_stack_records.append({
                'Method_1': name1, 'Method_2': name2,
                'AUC1 (95% CI)': auc1_str, 'AUC2 (95% CI)': auc2_str,
                'Z': z, 'P': p
            })
    df_delong_stack = pd.DataFrame(delong_stack_records)
    df_delong_stack = adjust_pvalues(df_delong_stack, p_col='P', method='fdr_bh')
    save_path_s15 = os.path.join(output_folder, '10_统计检验结果', '表S15_堆叠模型DeLong检验.csv')
    save_df(df_delong_stack, save_path_s15)
    print(f"[INFO] 已保存表S15")

    print("\n进行前5独立模型 vs 最佳堆叠模型 DeLong 检验...")
    best_stack_name = df_stack.iloc[0]['Combination']
    best_stack_prob = stacking_models[best_stack_name]['y_prob_test']

    delong_s16_records = []
    for name in top5_base:
        base_prob = base_models_trained[name].predict_proba(X_test)[:,1]
        z, p, auc_stack, auc_base = delong_test(y_test, best_stack_prob, base_prob)
        stack_auc_str = df_stack.iloc[0]['AUC (95% CI)']
        base_auc_str = df_base_final[df_base_final['Model']==name]['AUC (95% CI)'].values[0]
        delong_s16_records.append({
            'Method_1': best_stack_name, 'Method_2': name,
            'AUC1 (95% CI)': stack_auc_str, 'AUC2 (95% CI)': base_auc_str,
            'Z': z, 'P': p
        })
    df_delong_s16 = pd.DataFrame(delong_s16_records)
    df_delong_s16 = adjust_pvalues(df_delong_s16, p_col='P', method='fdr_bh')
    save_path_s16 = os.path.join(output_folder, '10_统计检验结果', '表S16_前5独立vs最佳堆叠DeLong检验.csv')
    save_df(df_delong_s16, save_path_s16)
    print(f"[INFO] 已保存表S16")

    print("\n生成综合排名（基础Top5 + 堆叠组合）...")
    base_top5_for_rank = df_base_final[df_base_final['Model'].isin(top5_base)].copy()
    base_top5_for_rank['Type'] = 'Base'
    base_top5_for_rank['Name'] = base_top5_for_rank['Model']

    stack_for_rank = df_stack.copy()
    stack_for_rank['Type'] = 'Stacking'
    stack_for_rank['Name'] = stack_for_rank['Combination']

    combined_df = pd.concat([base_top5_for_rank, stack_for_rank], ignore_index=True)

    combined_df, rank_cols_31 = comprehensive_ranking_with_ranks(combined_df, performance_metrics)

    for col in combined_df.columns:
        if combined_df[col].dtype in ['float64', 'float32']:
            combined_df[col] = combined_df[col].round(3)

    top5_combined = combined_df.head(5).reset_index(drop=True)

    for i, row in combined_df.iterrows():
        name = row['Name']
        if name not in MODEL_COLOR_MAP:
            if row['Type'] == 'Stacking':
                base_names = name.split('+')
                MODEL_COLOR_MAP[name] = get_stacking_color(base_names)
            else:
                MODEL_COLOR_MAP[name] = BASE_MODEL_COLORS.get(name, '#333333')
    for i, row in top5_combined.iterrows():
        MODEL_COLOR_MAP[row['Name']] = TOP5_COLORS[i]

    save_path_8b = os.path.join(output_folder, '3_综合排名与比较', '表8b_综合排名.csv')
    rank_display_cols_b = ['Name', 'Type', 'AUC (95% CI)', 'AUC', 'AUC_rank', 'Accuracy', 'Accuracy_rank',
                           'F1_score', 'F1_score_rank', 'G_mean', 'G_mean_rank',
                           'NPV', 'NPV_rank', 'PPV', 'PPV_rank', 'Sensitivity', 'Sensitivity_rank',
                           'Specificity', 'Specificity_rank', 'Comprehensive_Rank']
    rank_display_cols_b = [c for c in rank_display_cols_b if c in combined_df.columns]
    save_df(combined_df[rank_display_cols_b], save_path_8b)
    print("已保存表8b")

    print("\n进行综合前五模型 vs 基础前五模型 交叉 DeLong 检验...")

    def _get_prob_for_combined(name, mtype):
        if mtype == 'Base':
            return base_models_trained[name].predict_proba(X_test)[:,1]
        else:
            return stacking_models[name]['y_prob_test']

    delong_cross_records = []
    top5_names = top5_combined['Name'].tolist()
    top5_types = top5_combined['Type'].tolist()
    for i in range(len(top5_names)):
        for name_b in top5_base:
            name_a = top5_names[i]
            type_a = top5_types[i]
            if name_a == name_b:
                continue

            prob_a = _get_prob_for_combined(name_a, type_a)
            prob_b = base_models_trained[name_b].predict_proba(X_test)[:,1]

            z, p, auc_a, auc_b = delong_test(y_test, prob_a, prob_b)

            if type_a == 'Base':
                auc_a_str = df_base_final[df_base_final['Model']==name_a]['AUC (95% CI)'].values[0]
            else:
                auc_a_str = df_stack[df_stack['Combination']==name_a]['AUC (95% CI)'].values[0]
            auc_b_str = df_base_final[df_base_final['Model']==name_b]['AUC (95% CI)'].values[0]

            delong_cross_records.append({
                'Method_1': name_a, 'Method_2': name_b,
                'Type_1': type_a, 'Type_2': 'Base',
                'AUC1 (95% CI)': auc_a_str, 'AUC2 (95% CI)': auc_b_str,
                'Z': z, 'P': p
            })
    df_delong_cross = pd.DataFrame(delong_cross_records)
    df_delong_cross = adjust_pvalues(df_delong_cross, p_col='P', method='fdr_bh')
    save_path_s16c = os.path.join(output_folder, '10_统计检验结果', '表S16c_综合前5vs基础前5_DeLong检验.csv')
    save_df(df_delong_cross, save_path_s16c)
    print(f"[INFO] 已保存表S16c")

    print("\n生成完整可视化图表...")
    all_models_dict = {**base_models_trained}
    for name, info in stacking_models.items():
        all_models_dict[name] = StackingWrapper(info['base_models'], info['meta_model'], feature_names)

    metrics_for_radar = ['AUC','Accuracy','F1_score','Sensitivity','Specificity','PPV','NPV']
    base_roc_data = []
    base_pr_data = []
    base_cal_data = []

    for name, model in base_models_trained.items():
        safe_name = name.replace(' ', '_').replace('/', '_')
        color = BASE_MODEL_COLORS.get(name, '#333333')
        y_prob = model.predict_proba(X_test)[:,1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        base_roc_data.append((name, fpr, tpr, color))
        base_pr_data.append((name, y_prob, color))
        base_cal_data.append((name, y_prob, color))

        plot_roc_clean(fpr, tpr, color, name, os.path.join(output_folder, '4_ROC曲线', f'Base_{safe_name}_ROC.tiff'))
        plot_pr_clean(precision, recall, color, name, os.path.join(output_folder, '4_PR曲线', f'Base_{safe_name}_PR.tiff'), y_true=y_test, y_prob=y_prob)
        plot_dca_standard(y_test, y_prob, color, name, os.path.join(output_folder, '6_DCA曲线', f'Base_{safe_name}_DCA.tiff'))
        plot_calibration_curve(y_test, y_prob, color, name, os.path.join(output_folder, '7_校准曲线', f'Base_{safe_name}_Calibration.tiff'))
        row = df_base_final[df_base_final['Model']==name].iloc[0]
        metrics_raw = [row[m] for m in metrics_for_radar]
        angles = np.linspace(0, 2*np.pi, len(metrics_for_radar), endpoint=False).tolist()
        metrics_norm = [max(0, min(1, m)) for m in metrics_raw]
        metrics_norm += metrics_norm[:1]
        angles += angles[:1]
        plot_radar_clean(angles, metrics_norm, metrics_for_radar, color, name,
                        os.path.join(output_folder, '5_雷达图', f'Base_{safe_name}_Radar.tiff'))

    plot_combined_roc(base_roc_data, os.path.join(output_folder, '11_汇总可视化', '图1_12基础模型汇总ROC.tiff'),
                     'Combined ROC Curves - 12 Base Models')
    plot_combined_pr(base_pr_data, os.path.join(output_folder, '11_汇总可视化', '图1b_12基础模型汇总PR.tiff'),
                    'Combined PR Curves - 12 Base Models', y_true=y_test)
    plot_combined_calibration_curve(y_test, base_cal_data,
                                   os.path.join(output_folder, '11_汇总可视化', '图6_12基础模型汇总校准.tiff'),
                                   'Combined Calibration Curves - 12 Base Models')

    fig, ax = plt.subplots(figsize=(16, 12))
    thresholds = np.linspace(0.01, 0.99, 200)
    event_rate = np.mean(y_test)
    for name in base_models_trained.keys():
        y_prob = base_models_trained[name].predict_proba(X_test)[:,1]
        net_benefit = []
        for thresh in thresholds:
            pred_pos = (y_prob >= thresh).astype(int)
            tp = np.sum((pred_pos == 1) & (y_test == 1))
            fp = np.sum((pred_pos == 1) & (y_test == 0))
            nb = (tp/len(y_test)) - (fp/len(y_test)) * (thresh/(1-thresh)) if thresh < 1 else 0
            net_benefit.append(nb)
        net_benefit = np.array(net_benefit)
        ax.plot(thresholds, net_benefit, lw=2.5, color=BASE_MODEL_COLORS.get(name, '#333333'), label=name)
    nb_all = event_rate - (1 - event_rate) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, nb_all, 'k--', lw=2, label='All')
    ax.plot(thresholds, np.zeros_like(thresholds), 'k-', lw=2, label='None')
    ax.set_xlabel('Threshold Probability', fontsize=27, fontweight='bold')
    ax.set_ylabel('Net Benefit', fontsize=27, fontweight='bold')
    ax.set_title('Decision Curve Analysis (12 Base Models)', fontsize=33, fontweight='bold')
    ax.set_ylim(-0.1, 0.5)
    _set_bold_ticks(ax, 24)
    _set_spines(ax, 2.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '6_DCA曲线', 'DCA_12基础模型汇总.tiff'), dpi=300, format='tiff')
    plt.close()

    top5_roc_data = []
    top5_pr_data = []
    top5_cal_data = []

    for idx, row in top5_combined.iterrows():
        model_name = row['Name']
        model_type = row['Type']
        color = MODEL_COLOR_MAP[model_name]
        safe_name = model_name.replace(' ', '_').replace('+', '_').replace('/', '_')

        if model_type == 'Base':
            y_prob = base_models_trained[model_name].predict_proba(X_test)[:,1]
        else:
            y_prob = stacking_models[model_name]['y_prob_test']

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        top5_roc_data.append((model_name, fpr, tpr, color))
        top5_pr_data.append((model_name, y_prob, color))
        top5_cal_data.append((model_name, y_prob, color))

        plot_roc_clean(fpr, tpr, color, model_name, os.path.join(output_folder, '4_ROC曲线', f'Top5_{safe_name}_ROC.tiff'))
        plot_pr_clean(precision, recall, color, model_name, os.path.join(output_folder, '4_PR曲线', f'Top5_{safe_name}_PR.tiff'), y_true=y_test, y_prob=y_prob)
        plot_dca_standard(y_test, y_prob, color, model_name, os.path.join(output_folder, '6_DCA曲线', f'Top5_{safe_name}_DCA.tiff'))
        plot_calibration_curve(y_test, y_prob, color, model_name, os.path.join(output_folder, '7_校准曲线', f'Top5_{safe_name}_Calibration.tiff'))

        metrics_raw = [row[m] for m in metrics_for_radar]
        angles = np.linspace(0, 2*np.pi, len(metrics_for_radar), endpoint=False).tolist()
        metrics_norm = [max(0, min(1, m)) for m in metrics_raw]
        metrics_norm += metrics_norm[:1]
        angles += angles[:1]
        plot_radar_clean(angles, metrics_norm, metrics_for_radar, color, model_name,
                        os.path.join(output_folder, '5_雷达图', f'Top5_{safe_name}_Radar.tiff'))

    plot_combined_roc(top5_roc_data, os.path.join(output_folder, '11_汇总可视化', '图4_综合前5模型汇总ROC.tiff'),
                     'Combined ROC Curves - Top 5 Comprehensive Models')
    plot_combined_pr(top5_pr_data, os.path.join(output_folder, '11_汇总可视化', '图4b_综合前5模型汇总PR.tiff'),
                    'Combined PR Curves - Top 5 Comprehensive Models', y_true=y_test)
    plot_combined_calibration_curve(y_test, top5_cal_data,
                                   os.path.join(output_folder, '11_汇总可视化', '图6_综合前5模型汇总校准.tiff'),
                                   'Combined Calibration Curves - Top 5 Comprehensive Models')

    fig, ax = plt.subplots(figsize=(16, 12))
    for idx, row in top5_combined.iterrows():
        model_name = row['Name']
        model_type = row['Type']
        color = MODEL_COLOR_MAP[model_name]
        if model_type == 'Base':
            y_prob = base_models_trained[model_name].predict_proba(X_test)[:,1]
        else:
            y_prob = stacking_models[model_name]['y_prob_test']
        net_benefit = []
        for thresh in thresholds:
            pred_pos = (y_prob >= thresh).astype(int)
            tp = np.sum((pred_pos == 1) & (y_test == 1))
            fp = np.sum((pred_pos == 1) & (y_test == 0))
            nb = (tp/len(y_test)) - (fp/len(y_test)) * (thresh/(1-thresh)) if thresh < 1 else 0
            net_benefit.append(nb)
        ax.plot(thresholds, net_benefit, lw=3, color=color, label=model_name)
    nb_all = event_rate - (1 - event_rate) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, nb_all, 'k--', lw=2, label='All')
    ax.plot(thresholds, np.zeros_like(thresholds), 'k-', lw=2, label='None')
    ax.set_xlabel('Threshold Probability', fontsize=27, fontweight='bold')
    ax.set_ylabel('Net Benefit', fontsize=27, fontweight='bold')
    ax.set_title('Decision Curve Analysis (Top 5)', fontsize=33, fontweight='bold')
    ax.set_ylim(-0.1, 0.5)
    _set_bold_ticks(ax, 24)
    _set_spines(ax, 2.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '6_DCA曲线', 'DCA_Top5.tiff'), dpi=300, format='tiff')
    plt.close()

    base_rank_data = []
    for _, row in base_rank_df.iterrows():
        name = row['Model']
        score = row['Comprehensive_Rank']
        color = MODEL_COLOR_MAP.get(name, BASE_MODEL_COLORS.get(name, '#333333'))
        base_rank_data.append((name, score, color))
    plot_horizontal_bar(base_rank_data, os.path.join(output_folder, '3_综合排名与比较', 'Rank_12基础模型.tiff'),
                         'Comprehensive Rank of 12 Base Models')

    comb_rank_data = []
    for _, row in combined_df.iterrows():
        name = row['Name']
        score = row['Comprehensive_Rank']
        color = MODEL_COLOR_MAP[name]
        comb_rank_data.append((name, score, color))
    plot_horizontal_bar(comb_rank_data, os.path.join(output_folder, '3_综合排名与比较', 'Rank_综合模型.tiff'),
                         'Comprehensive Rank of Models (Top 5 Base + Stacking)')

    top5_comb_rank_data = []
    for _, row in top5_combined.iterrows():
        name = row['Name']
        score = row['Comprehensive_Rank']
        color = MODEL_COLOR_MAP[name]
        top5_comb_rank_data.append((name, score, color))
    plot_horizontal_bar(top5_comb_rank_data, os.path.join(output_folder, '3_综合排名与比较', 'Rank_综合前5模型.tiff'),
                         'Top 5 Comprehensive Models - Ranking')

    if SHAP_AVAILABLE:
        print("\n开始SHAP分析（综合排名前五模型）...")
        shap_output_folder = os.path.join(output_folder, '8_SHAP分析')

        global_max_shap = 0.0
        shap_data = []
        for idx, row in top5_combined.iterrows():
            model_name = row['Name']
            model_type = row['Type']
            color = MODEL_COLOR_MAP[model_name]
            print(f"  预计算SHAP [{idx+1}/5]: {model_name}")
            try:
                if model_type == 'Base':
                    model_obj = base_models_trained[model_name]
                else:
                    info = stacking_models[model_name]
                    class StackingWrapperSHAP:
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
                            for n, m in self.base_models.items():
                                meta_dict[n] = m.predict_proba(X_df)[:,1]
                            X_meta = pd.DataFrame(meta_dict)
                            return self.meta_model.predict_proba(X_meta)
                    wrapper = StackingWrapperSHAP(info['base_models'], info['meta_model'], feature_names)
                    model_obj = wrapper

                temp_analyzer = SHAPAnalyzer(X_test, feature_names,
                                            os.path.join(shap_output_folder, model_name.replace('+', '_')),
                                            model_color=color)
                shap_values = temp_analyzer._compute_shap_values(model_obj)
                if shap_values is not None:
                    max_abs = float(np.abs(shap_values).max())
                    global_max_shap = max(global_max_shap, max_abs)
                    shap_data.append((model_name, model_type, color, model_obj, shap_values))
            except Exception as e:
                print(f"    SHAP预计算失败: {str(e)[:100]}")

        if shap_data:
            global_xlim_val = global_max_shap * 1.15
            print(f"  全局SHAP最大绝对值: {global_max_shap:.3f}, 统一xlim: {global_xlim_val:.3f}")
            for model_name, model_type, color, model_obj, shap_values in shap_data:
                print(f"  绘制SHAP图: {model_name}")
                try:
                    analyzer = SHAPAnalyzer(X_test, feature_names,
                                           os.path.join(shap_output_folder, model_name.replace('+', '_')),
                                           model_color=color, global_xlim=global_xlim_val)
                    analyzer._plot_shap_summary_bar(model_name, shap_values)
                    analyzer._plot_shap_beeswarm(model_name, shap_values)
                    print(f"    {model_name} SHAP分析完成")
                except Exception as e:
                    print(f"    SHAP绘图失败: {str(e)[:100]}")

    print("\n保存综合前五模型到专用文件夹...")
    save_date = datetime.now().strftime("%Y%m%d")
    model_output_folder = f"{save_date}_输出模型"
    os.makedirs(model_output_folder, exist_ok=True)

    feature_df = pd.DataFrame({'feature_name': feature_names})
    feature_df.to_csv(os.path.join(model_output_folder, 'feature_names.csv'), index=False, encoding='utf-8-sig')
    joblib.dump(scaler, os.path.join(model_output_folder, 'scaler.joblib'))

    for idx, row in top5_combined.iterrows():
        model_name = row['Name']
        model_type = row['Type']
        safe_name = model_name.replace('+', '_').replace(' ', '_').replace('/', '_')
        if model_type == 'Base':
            model_obj = base_models_trained[model_name]
            threshold = model_thresholds.get(model_name, 0.5)
            model_data = {
                'model': model_obj, 'name': model_name, 'type': 'Base', 'threshold': threshold,
                'features': feature_names,
                'performance': {
                    'AUC': float(row['AUC']), 'AUC_CI': row['AUC (95% CI)'],
                    'Accuracy': float(row['Accuracy']), 'F1_score': float(row['F1_score']),
                    'Sensitivity': float(row['Sensitivity']), 'Specificity': float(row['Specificity'])
                }
            }
            config = {
                'model_name': model_name, 'model_type': 'Base', 'rank': int(idx+1),
                'threshold': float(threshold), 'features': feature_names,
                'note': '基于核心特征训练，外部验证需提供相同特征并做相同标准化',
                'performance': model_data['performance']
            }
        else:
            info = stacking_models[model_name]
            threshold = model_thresholds.get(model_name, 0.5)
            model_data = {
                'meta_model': info['meta_model'], 'base_models': info['base_models'],
                'combination': info['combination'], 'name': model_name, 'type': 'Stacking',
                'features': feature_names, 'threshold': threshold,
                'performance': {
                    'AUC': float(row['AUC']), 'AUC_CI': row['AUC (95% CI)'],
                    'Accuracy': float(row['Accuracy']), 'F1_score': float(row['F1_score']),
                    'Sensitivity': float(row['Sensitivity']), 'Specificity': float(row['Specificity'])
                }
            }
            config = {
                'model_name': model_name, 'model_type': 'Stacking', 'rank': int(idx+1),
                'base_models': list(info['combination']), 'features': feature_names,
                'threshold': float(threshold),
                'note': '需要先获取基础模型预测作为元特征，再输入meta_model',
                'performance': model_data['performance']
            }
        joblib.dump(model_data, os.path.join(model_output_folder, f'{idx+1}_{safe_name}_model.joblib'))
        with open(os.path.join(model_output_folder, f'{idx+1}_{safe_name}_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"  已保存模型 [{idx+1}/5]: {model_name}")

    readme_lines = [
        "# 综合前五模型使用说明", f"生成日期：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        "## 模型列表（按综合排名）"
    ]
    for idx, row in top5_combined.iterrows():
        readme_lines.append(f"{idx+1}. **{row['Name']}** (AUC: {row['AUC']}, 类型: {row['Type']})")
    readme_lines.extend([
        "", "## 文件说明", "- `*_model.joblib`: 模型文件", "- `*_config.json`: 模型配置信息",
        "- `feature_names.csv`: 核心特征名称列表（必须严格按此顺序输入）",
        "- `scaler.joblib`: 标准化器（需对输入数据做相同标准化）",
        "", "## 使用方法", "```python", "import joblib, pandas as pd, numpy as np",
        "from sklearn.preprocessing import StandardScaler", "",
        "# 加载模型", "model_data = joblib.load('1_xxx_model.joblib')",
        "scaler = joblib.load('scaler.joblib')",
        "features = pd.read_csv('feature_names.csv')['feature_name'].tolist()", "",
        "# 准备外部数据", "X_external = pd.read_csv('your_data.csv')[features]",
        "X_scaled = scaler.transform(X_external)", "",
        "# 预测", "y_prob = model_data['model'].predict_proba(X_scaled)[:, 1]",
        "y_pred = (y_prob >= model_data['threshold']).astype(int)", "```"
    ])
    with open(os.path.join(model_output_folder, 'README.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(readme_lines))

    print(f"\n模型已保存至文件夹：{model_output_folder}")
    print(f"使用说明：{os.path.join(model_output_folder, 'README.md')}")
    print("\n[INFO] 代码包3 全部完成")
    print(f"输出文件夹: {os.path.abspath(output_folder)}")
    print("=" * 60)

if __name__ == '__main__':
    main()
