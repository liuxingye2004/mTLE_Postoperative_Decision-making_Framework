#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
额叶癫痫术后复发风险分层 - Cox回归完整验证优化版（修复版）

功能：
1. 使用建模队列构建Cox回归模型
2. 使用X-tile确定最佳截断值
3. 在验证队列中评估模型性能
4. 展示关键指标：HR值、C-index、Brier Score、KM生存曲线
5. 对验证队列进行风险分层
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import chi2_contingency
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
import seaborn as sns
from itertools import combinations
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置字体
def setup_fonts():
    import matplotlib.font_manager as fm
    font_candidates = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
    if selected_font:
        plt.rcParams['font.family'] = [selected_font, 'DejaVu Sans']
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    return selected_font

setup_fonts()

# 数据预处理模块
def load_and_preprocess_data(filepath, handle_missing='impute'):
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath, encoding='utf-8')
    
    df.columns = df.columns.str.strip()
    print("=" * 80)
    print(f"数据加载: {os.path.basename(filepath)}")
    print("=" * 80)
    print(f"原始数据维度: {df.shape}")
    
    print("\n缺失值诊断:")
    print("-" * 50)
    missing_info = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'缺失数量': missing_info, '缺失百分比': missing_percent})
    print(missing_df[missing_df['缺失数量'] > 0])
    
    total_missing_rows = df.isnull().any(axis=1).sum()
    print(f"\n至少有一个缺失值的患者: {total_missing_rows} 例 ({total_missing_rows/len(df)*100:.1f}%)")
    
    feature_cols = ['onset age', 'duration', 'frequency', 'Interictal EEG', 'LYMPH', 
                   'ALT', 'MRI', 'PETSUVmax', 'AI']
    
    if handle_missing == 'drop':
        df_clean = df.dropna()
        print(f"\n删除缺失值后: {len(df_clean)} 例")
    elif handle_missing == 'impute':
        print(f"\n使用中位数填充缺失值...")
        df_clean = df.copy()
        imputable_cols = [c for c in feature_cols if c in df_clean.columns 
                         and df_clean[c].dtype.kind in 'fc']
        if imputable_cols:
            imputer = SimpleImputer(strategy='median')
            df_clean[imputable_cols] = imputer.fit_transform(df_clean[imputable_cols])
        categorical_cols = [c for c in feature_cols if c in df_clean.columns 
                           and c not in imputable_cols]
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        print(f"填充后样本量: {len(df_clean)} 例")
    else:
        df_clean = df.copy()
    
    if 'FUFA' in df_clean.columns:
        recurrence_rate = df_clean['FUFA'].mean()
        print(f"\n复发情况: {df_clean['FUFA'].sum()}/{len(df_clean)} ({recurrence_rate*100:.1f}%)")
    
    if 'time' in df_clean.columns:
        print(f"随访时间统计: 中位数={df_clean['time'].median():.1f}月, "
              f"范围=[{df_clean['time'].min():.1f}, {df_clean['time'].max():.1f}]月")
        invalid_time = (df_clean['time'] <= 0).sum()
        if invalid_time > 0:
            print(f"⚠ 排除 {invalid_time} 例随访时间无效的患者")
            df_clean = df_clean[df_clean['time'] > 0]
    
    df_clean = df_clean.reset_index(drop=True)
    print(f"最终有效样本量: {len(df_clean)} 例")
    return df_clean, len(df)

# Cox比例风险模型类
class RiskPredictionModel:
    def __init__(self):
        self.coefficients = {}
        self.features = None
        self.model = None
        self.scaler = None
        self.median_fu = None
        self.feature_cols = None
        self.duration_col = None
        self.event_col = None
        self.baseline_hazard = None
        self.c_index = None
        self.log_likelihood_ = None
        self.AIC_partial_ = None
    
    def _get_feature_df(self, df):
        X = df[self.feature_cols].copy()
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.feature_cols, index=X.index)
    
    def fit(self, df, feature_cols, duration_col='time', event_col='FUFA'):
        self.feature_cols = feature_cols
        self.duration_col = duration_col
        self.event_col = event_col
        
        required_cols = feature_cols + [duration_col, event_col]
        df_model = df[required_cols].copy()
        
        self.median_fu = df_model[duration_col].median()
        
        print("\n" + "=" * 80)
        print("Cox比例风险模型结果 (建模队列)")
        print("=" * 80)
        print(f"中位随访时间: {self.median_fu:.1f} 个月")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(df_model[feature_cols])
        df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df_model.index)
        df_scaled[duration_col] = df_model[duration_col].values
        df_scaled[event_col] = df_model[event_col].values
        
        self.model = CoxPHFitter(penalizer=0.0)
        self.model.fit(df_scaled, duration_col=duration_col, event_col=event_col)
        
        self.coefficients = dict(zip(feature_cols, self.model.params_.values))
        self.c_index = self.model.concordance_index_
        self.log_likelihood_ = self.model.log_likelihood_
        self.AIC_partial_ = self.model.AIC_partial_
        
        print(f"\nConcordance Index (C-index): {self.c_index:.3f}")
        print(f"Log-Likelihood: {self.log_likelihood_:.2f}")
        print(f"AIC: {self.AIC_partial_:.2f}")
        
        print("\n风险因素重要性 (风险比 HR):")
        print("-" * 80)
        print(f"{'特征':<20} {'HR':<10} {'系数':<10} {'P值':<15} {'显著性'}")
        print("-" * 80)
        sorted_coef = sorted(self.coefficients.items(), 
                            key=lambda x: abs(x[1]), reverse=True)
        for feat, coef in sorted_coef:
            hr = np.exp(coef)
            p_val = self.model.summary.loc[feat, 'p'] if feat in self.model.summary.index else np.nan
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"{feat:<20} {hr:<10.3f} {coef:<10.3f} {p_val:<15.4f} {sig}")
        
        print("\n*** p<0.001, ** p<0.01, * p<0.05")
        return self
    
    def predict_risk_score(self, df):
        df_features = self._get_feature_df(df)
        surv_funcs = self.model.predict_survival_function(df_features)
        
        if self.median_fu in surv_funcs.index:
            surv_at_median = surv_funcs.loc[self.median_fu].values
        else:
            surv_at_median = np.array([
                np.interp(self.median_fu, surv_funcs.index, surv_funcs.iloc[:, i].values)
                for i in range(surv_funcs.shape[1])
            ])
        
        risk_probs = 1.0 - surv_at_median
        return np.clip(risk_probs, 0.0, 1.0)
    
    def get_linear_predictor(self, df):
        df_features = self._get_feature_df(df)
        return self.model.predict_log_partial_hazard(df_features).to_numpy()
    
    def get_partial_hazard(self, df):
        return np.exp(self.get_linear_predictor(df))

# X-tile优化模块
def x_tile_optimization(df, risk_scores, outcome_col='FUFA', 
                       min_group_size=0.10, min_events_per_group=3):
    print("\n" + "=" * 80)
    print("X-tile优化分析 (基于建模队列)")
    print("=" * 80)
    
    events = df[outcome_col].values
    assert len(risk_scores) == len(df), f"风险评分数量与患者数量不匹配!"
    print(f"输入患者数: {len(df)}")
    
    q5, q95 = np.percentile(risk_scores, [5, 95])
    candidate_cutoffs = np.linspace(q5, q95, 50)
    
    results = []
    best_diff = -1
    best_cutoffs = None
    
    print("\n扫描最优截断值...")
    
    for i, cutoff1 in enumerate(candidate_cutoffs):
        for cutoff2 in candidate_cutoffs[i+1:]:
            if cutoff2 - cutoff1 < 0.02:
                continue
            
            groups = np.where(risk_scores <= cutoff1, 0,
                     np.where(risk_scores <= cutoff2, 1, 2))
            
            unique, counts = np.unique(groups, return_counts=True)
            if len(unique) < 3:
                continue
            
            min_count = min(counts)
            if min_count < len(df) * min_group_size:
                continue
            
            valid = True
            for g in [0, 1, 2]:
                mask = groups == g
                if np.sum(events[mask] == 1) < min_events_per_group:
                    valid = False
                    break
            
            if not valid:
                continue
            
            rates = [np.mean(events[groups == g]) * 100 for g in [0, 1, 2]]
            
            if not (rates[0] < rates[1] < rates[2]):
                continue
            
            rate_diff = rates[2] - rates[0]
            
            results.append({
                'cutoff1': cutoff1,
                'cutoff2': cutoff2,
                'low_rate': rates[0],
                'med_rate': rates[1],
                'high_rate': rates[2],
                'rate_diff': rate_diff,
                'low_n': counts[0],
                'med_n': counts[1],
                'high_n': counts[2]
            })
            
            if rate_diff > best_diff:
                best_diff = rate_diff
                best_cutoffs = (cutoff1, cutoff2)
    
    if best_cutoffs is None:
        print("标准较严，放宽条件重新搜索...")
        return x_tile_optimization(df, risk_scores, outcome_col, 
                                   min_group_size=0.05, min_events_per_group=2)
    
    results_df = pd.DataFrame(results)
    
    print("\n✓ 最优截断值:")
    print(f"  低危 ≤ {best_cutoffs[0]:.3f} ({best_cutoffs[0]*100:.1f}%)")
    print(f"  中危 ≤ {best_cutoffs[1]:.3f} ({best_cutoffs[1]*100:.1f}%)")
    print(f"  高危 > {best_cutoffs[1]:.3f}")
    print(f"  高低危复发率差异: {best_diff:.1f}%")
    
    return results_df, best_cutoffs

# 模型验证器类 (优化版)
class ModelValidator:
    def __init__(self, risk_model, duration_col='time', event_col='FUFA'):
        self.risk_model = risk_model
        self.duration_col = duration_col
        self.event_col = event_col
    
    def calculate_c_index(self, df, risk_scores=None):
        if risk_scores is None:
            risk_scores = self.risk_model.predict_risk_score(df)
        
        c_index = concordance_index(
            df[self.duration_col].values,
            -risk_scores,  # 注意：lifelines的concordance_index期望风险高的数值小
            df[self.event_col].values
        )
        return c_index
    
    def bootstrap_c_index(self, df, risk_scores=None, n_bootstrap=1000, random_state=42):
        np.random.seed(random_state)
        n_samples = len(df)
        
        if risk_scores is None:
            risk_scores = self.risk_model.predict_risk_score(df)
        
        base_c_index = self.calculate_c_index(df, risk_scores)
        
        bootstrap_c_indices = []
        for i in range(n_bootstrap):
            boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_df = df.iloc[boot_indices].reset_index(drop=True)
            boot_scores = risk_scores[boot_indices]
            
            try:
                c_idx = self.calculate_c_index(boot_df, boot_scores)
                bootstrap_c_indices.append(c_idx)
            except Exception:
                continue
        
        bootstrap_c_indices = np.array(bootstrap_c_indices)
        
        return {
            'c_index': base_c_index,
            'ci_lower': np.percentile(bootstrap_c_indices, 2.5),
            'ci_upper': np.percentile(bootstrap_c_indices, 97.5),
            'bootstrap_mean': np.mean(bootstrap_c_indices),
            'bootstrap_std': np.std(bootstrap_c_indices)
        }
    
    def calculate_brier_score(self, df, eval_times=None):
        if eval_times is None:
            eval_times = [12, 24, 36]
        
        surv_probs_df = self._predict_survival_probabilities(df, eval_times)
        brier_scores = {}
        
        for t in eval_times:
            col = f'{t}mo'
            pred_surv = surv_probs_df[col].values
            
            mask_event = (df[self.duration_col] <= t) & (df[self.event_col] == 1)
            mask_censored_before = (df[self.duration_col] <= t) & (df[self.event_col] == 0)
            
            observed = np.zeros(len(df))
            observed[mask_event] = 1.0
            observed[mask_censored_before] = np.nan
            
            valid_mask = ~np.isnan(observed)
            if np.sum(valid_mask) > 0:
                brier = np.mean((pred_surv[valid_mask] - (1 - observed[valid_mask])) ** 2)
            else:
                brier = np.nan
            
            brier_scores[col] = brier
        
        valid_scores = [v for v in brier_scores.values() if not np.isnan(v)]
        brier_scores['mean'] = np.mean(valid_scores) if valid_scores else np.nan
        return brier_scores
    
    def _predict_survival_probabilities(self, df, times):
        df_features = self.risk_model._get_feature_df(df)
        surv_funcs = self.risk_model.model.predict_survival_function(df_features)
        
        probs = {}
        for t in times:
            col = f'{t}mo'
            if t in surv_funcs.index:
                probs[col] = surv_funcs.loc[t].values
            else:
                probs[col] = np.array([
                    np.interp(t, surv_funcs.index, surv_funcs.iloc[:, i].values)
                    for i in range(surv_funcs.shape[1])
                ])
        
        return pd.DataFrame(probs, index=df.index)
    
    def validate(self, train_df, valid_df=None, eval_times=[12, 24, 36]):
        print("\n" + "=" * 80)
        print("模型验证分析")
        print("=" * 80)
        
        train_risk_scores = self.risk_model.predict_risk_score(train_df)
        
        results = {}
        
        print("\n" + "=" * 80)
        print("1. 建模队列评估")
        print("=" * 80)
        
        train_cindex = self.bootstrap_c_index(train_df, train_risk_scores)
        print(f"\nC-index: {train_cindex['c_index']:.3f} (95% CI: {train_cindex['ci_lower']:.3f}-{train_cindex['ci_upper']:.3f})")
        
        train_brier = self.calculate_brier_score(train_df, eval_times)
        print(f"\nBrier Score (均值): {train_brier['mean']:.4f}")
        for t in eval_times:
            col = f'{t}mo'
            if col in train_brier and not np.isnan(train_brier[col]):
                print(f"  {t}个月: {train_brier[col]:.4f}")
        
        results['train'] = {
            'cindex': train_cindex,
            'brier': train_brier,
            'risk_scores': train_risk_scores
        }
        
        if valid_df is not None:
            print("\n" + "=" * 80)
            print("2. 验证队列评估")
            print("=" * 80)
            
            valid_risk_scores = self.risk_model.predict_risk_score(valid_df)
            
            valid_cindex = self.bootstrap_c_index(valid_df, valid_risk_scores)
            print(f"\nC-index: {valid_cindex['c_index']:.3f} (95% CI: {valid_cindex['ci_lower']:.3f}-{valid_cindex['ci_upper']:.3f})")
            
            valid_brier = self.calculate_brier_score(valid_df, eval_times)
            print(f"\nBrier Score (均值): {valid_brier['mean']:.4f}")
            for t in eval_times:
                col = f'{t}mo'
                if col in valid_brier and not np.isnan(valid_brier[col]):
                    print(f"  {t}个月: {valid_brier[col]:.4f}")
            
            results['valid'] = {
                'cindex': valid_cindex,
                'brier': valid_brier,
                'risk_scores': valid_risk_scores
            }
        
        return results

# 可视化模块 - 单独生成每个图
def plot_kaplan_meier_single(df, groups, title, suffix, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    group_configs = [
        ('低危', '#2ecc71', 'Low-risk'),
        ('中危', '#f39c12', 'Moderate-risk'),
        ('高危', '#e74c3c', 'High-risk'),
    ]
    
    kmfitters = []
    for group_name, color, label_en in group_configs:
        mask = groups == group_name
        if np.sum(mask) == 0:
            continue
        
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, 'time'], event_observed=df.loc[mask, 'FUFA'], label=label_en)
        kmfitters.append((kmf, color))
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2)
    
    ax.set_title(f'{title} - Kaplan-Meier生存曲线', fontsize=14, fontweight='bold')
    ax.set_xlabel('随访时间 (月)', fontsize=12)
    ax.set_ylabel('无复发生存率', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_KM_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {title}KM生存曲线已保存: 01_KM_{suffix}.png")

def plot_comparison_kaplan_meier(train_df, valid_df, train_groups, valid_groups, 
                                  best_cutoffs, output_dir):
    plot_kaplan_meier_single(train_df, train_groups, '建模队列', 'train', output_dir)
    plot_kaplan_meier_single(valid_df, valid_groups, '验证队列', 'valid', output_dir)

def plot_metrics_single(results, which, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = ['C-index', 'Brier Score (均值)']
    values = [
        results[which]['cindex']['c_index'],
        results[which]['brier']['mean']
    ]
    
    x = np.arange(len(metrics))
    width = 0.6
    
    ax1 = axes[0]
    bars1 = ax1.bar(x, values, width, label=which, color='#3498db' if which == 'train' else '#e74c3c', alpha=0.8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.set_title(f'{which}模型评估指标', fontsize=14, fontweight='bold')
    ax1.set_ylabel('数值', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    ax2 = axes[1]
    eval_times = [12, 24, 36]
    briers = [results[which]['brier'][f'{t}mo'] for t in eval_times]
    
    x_brier = np.arange(len(eval_times))
    ax2.plot(x_brier, briers, 'o-', linewidth=3, markersize=10, 
             label=which, color='#3498db' if which == 'train' else '#e74c3c')
    
    ax2.set_xticks(x_brier)
    ax2.set_xticklabels([f'{t}个月' for t in eval_times], fontsize=11)
    ax2.set_title(f'{which}Brier Score随时间变化', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Brier Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for i, v in enumerate(briers):
        ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_metrics_{which}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {which}指标图已保存: 02_metrics_{which}.png")

def plot_metrics_comparison(results, output_dir):
    plot_metrics_single(results, 'train', output_dir)
    plot_metrics_single(results, 'valid', output_dir)

def plot_risk_distribution_single(df, scores, title, suffix, best_cutoffs, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    events = df['FUFA'].values
    ax.hist(scores[events == 0], bins=30, alpha=0.6, 
            label='未复发', color='#2ecc71', density=True, edgecolor='black')
    ax.hist(scores[events == 1], bins=30, alpha=0.6, 
            label='复发', color='#e74c3c', density=True, edgecolor='black')
    ax.axvline(best_cutoffs[0], color='black', linestyle='--', linewidth=2, 
               label=f'低危/中危: {best_cutoffs[0]:.3f}')
    ax.axvline(best_cutoffs[1], color='black', linestyle='--', linewidth=2, 
               label=f'中危/高危: {best_cutoffs[1]:.3f}')
    ax.set_xlabel('风险评分', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title(f'{title} - 风险评分分布', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_risk_distribution_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {title}风险评分分布图已保存: 03_risk_distribution_{suffix}.png")

def plot_risk_distribution(train_df, valid_df, train_scores, valid_scores, 
                          best_cutoffs, output_dir):
    plot_risk_distribution_single(train_df, train_scores, '建模队列', 'train', best_cutoffs, output_dir)
    plot_risk_distribution_single(valid_df, valid_scores, '验证队列', 'valid', best_cutoffs, output_dir)

def plot_risk_groups_single(df, groups, title, suffix, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    group_names = ['低危', '中危', '高危']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    group_stats = []
    for name in group_names:
        mask = groups == name
        n = np.sum(mask)
        n_event = np.sum(df.loc[mask, 'FUFA'])
        rate = n_event / n * 100 if n > 0 else 0
        group_stats.append({
            'name': name,
            'n': n,
            'n_event': n_event,
            'rate': rate
        })
    
    ax1 = axes[0]
    bars = ax1.bar([s['name'] for s in group_stats], [s['n'] for s in group_stats], 
                  color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title(f'{title} - 各风险组人数', fontsize=13, fontweight='bold')
    ax1.set_ylabel('人数', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, 
                f'{group_stats[i]["n"]} ({group_stats[i]["n_event"]})', 
                ha='center', va='bottom', fontweight='bold')
    
    ax2 = axes[1]
    bars2 = ax2.bar([s['name'] for s in group_stats], [s['rate'] for s in group_stats], 
                   color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title(f'{title} - 各风险组复发率', fontsize=13, fontweight='bold')
    ax2.set_ylabel('复发率 (%)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, 
                f'{group_stats[i]["rate"]:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_risk_groups_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {title}风险组总结图已保存: 04_risk_groups_{suffix}.png")

def plot_risk_groups_summary(train_df, valid_df, train_groups, valid_groups, output_dir):
    plot_risk_groups_single(train_df, train_groups, '建模队列', 'train', output_dir)
    plot_risk_groups_single(valid_df, valid_groups, '验证队列', 'valid', output_dir)

# 风险分层函数
def apply_risk_stratification(df, risk_scores, best_cutoffs):
    groups = np.where(risk_scores <= best_cutoffs[0], '低危',
             np.where(risk_scores <= best_cutoffs[1], '中危', '高危'))
    return groups

# 主函数
def main_complete(train_file, valid_file, output_dir=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    if output_dir is None:
        output_dir = f'cox_validation_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("额叶癫痫术后复发风险分层 - 完整验证流程")
    print("=" * 80)
    
    feature_cols = ['onset age', 'duration', 'frequency', 'Interictal EEG', 'LYMPH', 
                   'ALT', 'MRI', 'PETSUVmax', 'AI']
    
    print(f"\n输出目录: {output_dir}")
    
    train_df, _ = load_and_preprocess_data(train_file, handle_missing='impute')
    valid_df, _ = load_and_preprocess_data(valid_file, handle_missing='impute')
    
    print("\n" + "=" * 80)
    print("步骤 1: 在建模队列中训练Cox模型")
    model = RiskPredictionModel()
    model.fit(train_df, feature_cols, duration_col='time', event_col='FUFA')
    
    print("\n" + "=" * 80)
    print("步骤 2: 在建模队列中进行X-tile优化确定截断值")
    train_risk_scores = model.predict_risk_score(train_df)
    _, best_cutoffs = x_tile_optimization(train_df, train_risk_scores)
    
    print("\n" + "=" * 80)
    print("步骤 3: 模型验证")
    validator = ModelValidator(model)
    results = validator.validate(train_df, valid_df)
    
    print("\n" + "=" * 80)
    print("步骤 4: 风险分层")
    print("=" * 80)
    
    train_groups = apply_risk_stratification(train_df, train_risk_scores, best_cutoffs)
    valid_risk_scores = model.predict_risk_score(valid_df)
    valid_groups = apply_risk_stratification(valid_df, valid_risk_scores, best_cutoffs)
    
    for name, df, groups in [('建模队列', train_df, train_groups), 
                            ('验证队列', valid_df, valid_groups)]:
        print(f"\n{name}风险分层结果:")
        print("-" * 50)
        for g in ['低危', '中危', '高危']:
            mask = groups == g
            n = np.sum(mask)
            n_event = np.sum(df.loc[mask, 'FUFA'])
            rate = n_event / n * 100 if n > 0 else 0
            print(f"  {g}: {n} 例, 复发 {n_event} 例, 复发率 {rate:.1f}%")
    
    print("\n" + "=" * 80)
    print("步骤 5: Log-rank检验")
    print("=" * 80)
    
    for name, df, groups in [('建模队列', train_df, train_groups), 
                            ('验证队列', valid_df, valid_groups)]:
        print(f"\n{name} Log-rank检验:")
        group_numeric = pd.Categorical(groups, categories=['低危', '中危', '高危']).codes
        try:
            lr_result = multivariate_logrank_test(df['time'], group_numeric, df['FUFA'])
            print(f"  三组整体: χ² = {lr_result.test_statistic:.2f}, p = {lr_result.p_value:.6f}")
            
            group_pairs = [('低危', '中危'), ('低危', '高危'), ('中危', '高危')]
            for g1, g2 in group_pairs:
                mask1 = groups == g1
                mask2 = groups == g2
                lr_pair = logrank_test(df.loc[mask1, 'time'], df.loc[mask2, 'time'],
                                      df.loc[mask1, 'FUFA'], df.loc[mask2, 'FUFA'])
                sig = '***' if lr_pair.p_value < 0.001 else '**' if lr_pair.p_value < 0.01 else '*' if lr_pair.p_value < 0.05 else ''
                print(f"  {g1} vs {g2}: χ² = {lr_pair.test_statistic:.2f}, p = {lr_pair.p_value:.4f} {sig}")
        except Exception as e:
            print(f"  计算失败: {e}")
    
    print("\n" + "=" * 80)
    print("步骤 6: 生成可视化结果")
    print("=" * 80)
    
    plot_comparison_kaplan_meier(train_df, valid_df, train_groups, valid_groups, 
                                  best_cutoffs, output_dir)
    plot_metrics_comparison(results, output_dir)
    plot_risk_distribution(train_df, valid_df, train_risk_scores, valid_risk_scores, 
                          best_cutoffs, output_dir)
    plot_risk_groups_summary(train_df, valid_df, train_groups, valid_groups, output_dir)
    
    print("\n" + "=" * 80)
    print("步骤 7: 保存分层结果")
    print("=" * 80)
    
    train_df['risk_score'] = train_risk_scores
    train_df['risk_group'] = train_groups
    train_df.to_csv(f'{output_dir}/training_set_with_risk.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 建模队列分层结果已保存: training_set_with_risk.csv")
    
    valid_df['risk_score'] = valid_risk_scores
    valid_df['risk_group'] = valid_groups
    valid_df.to_csv(f'{output_dir}/validation_set_with_risk.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 验证队列分层结果已保存: validation_set_with_risk.csv")
    
    summary_data = {
        '指标': ['C-index (建模)', 'C-index (验证)', 
                'C-index 95% CI (建模)', 'C-index 95% CI (验证)',
                'Brier Score均值 (建模)', 'Brier Score均值 (验证)',
                '低危截断值', '中危截断值'],
        '数值': [
            f"{results['train']['cindex']['c_index']:.3f}",
            f"{results['valid']['cindex']['c_index']:.3f}" if 'valid' in results else 'N/A',
            f"{results['train']['cindex']['ci_lower']:.3f}-{results['train']['cindex']['ci_upper']:.3f}",
            f"{results['valid']['cindex']['ci_lower']:.3f}-{results['valid']['cindex']['ci_upper']:.3f}" if 'valid' in results else 'N/A',
            f"{results['train']['brier']['mean']:.4f}",
            f"{results['valid']['brier']['mean']:.4f}" if 'valid' in results else 'N/A',
            f"{best_cutoffs[0]:.3f}",
            f"{best_cutoffs[1]:.3f}"
        ]
    }
    pd.DataFrame(summary_data).to_csv(f'{output_dir}/summary_results.csv', index=False, encoding='utf-8-sig')
    print("✓ 总结结果已保存: summary_results.csv")
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)
    
    return {
        'model': model,
        'best_cutoffs': best_cutoffs,
        'results': results,
        'train_df': train_df,
        'valid_df': valid_df,
        'train_groups': train_groups,
        'valid_groups': valid_groups
    }

if __name__ == "__main__":
    # 运行完整分析
    result = main_complete(
        train_file='4C_co1.csv',
        valid_file='4C_co2.csv',
        output_dir='cox_validation_complete'
    )
