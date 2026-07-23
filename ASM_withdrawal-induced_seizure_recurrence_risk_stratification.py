
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


def load_and_preprocess_data(filepath, handle_missing='impute'):
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath, encoding='utf-8')
    
    df.columns = df.columns.str.strip()
    print("=" * 80)
    print(f": {os.path.basename(filepath)}")
    print("=" * 80)
    print(f": {df.shape}")
    
    print("\n缺失值诊断:")
    print("-" * 50)
    missing_info = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'missing count': missing_info, 'missing persentage': missing_percent})
    print(missing_df[missing_df['missing conut'] > 0])
    
    total_missing_rows = df.isnull().any(axis=1).sum()
    print(f"\n: {total_missing_rows}  ({total_missing_rows/len(df)*100:.1f}%)")
    
    feature_cols = ['onset age', 'duration', 'frequency', 'Interictal EEG', 'LYMPH', 
                   'ALT', 'MRI', 'PETSUVmax', 'AI', 'R0 Resection', 'Abnormal Post-EEG', 'Seizure in 6m']
    
    if handle_missing == 'drop':
        df_clean = df.dropna()
        print(f"\n: {len(df_clean)} ")
    elif handle_missing == 'impute':
        print(f"\n...")
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
        print(f": {len(df_clean)} 例")
    else:
        df_clean = df.copy()
    
    if 'FUFA' in df_clean.columns:
        recurrence_rate = df_clean['FUFA'].mean()
        print(f"\n: {df_clean['FUFA'].sum()}/{len(df_clean)} ({recurrence_rate*100:.1f}%)")
    
    if 'time' in df_clean.columns:
        print(f": ={df_clean['time'].median():.1f}月, "
              f"=[{df_clean['time'].min():.1f}, {df_clean['time'].max():.1f}]月")
        invalid_time = (df_clean['time'] <= 0).sum()
        if invalid_time > 0:
            print(f"⚠  {invalid_time} ")
            df_clean = df_clean[df_clean['time'] > 0]
    
    df_clean = df_clean.reset_index(drop=True)
    print(f": {len(df_clean)} 例")
    return df_clean, len(df)


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
        print("Cox")
        print("=" * 80)
        print(f": {self.median_fu:.1f} ")
        
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
        
        print("\n:")
        print("-" * 80)
        print(f"{'Variables':<25} {'HR':<10} {'Coefficients':<10} {'P Value':<15} {'Significance'}")
        print("-" * 80)
        sorted_coef = sorted(self.coefficients.items(), 
                            key=lambda x: abs(x[1]), reverse=True)
        for feat, coef in sorted_coef:
            hr = np.exp(coef)
            p_val = self.model.summary.loc[feat, 'p'] if feat in self.model.summary.index else np.nan
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"{feat:<25} {hr:<10.3f} {coef:<10.3f} {p_val:<15.4f} {sig}")
        
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


def x_tile_optimization(df, risk_scores, outcome_col='FUFA', 
                       min_group_size=0.10, min_events_per_group=3):
    print("\n" + "=" * 80)
    print("X-tile")
    print("=" * 80)
    
    events = df[outcome_col].values
    assert len(risk_scores) == len(df), f"!"
    print(f": {len(df)}")
    
    q5, q95 = np.percentile(risk_scores, [5, 95])
    candidate_cutoffs = np.linspace(q5, q95, 50)
    
    results = []
    best_diff = -1
    best_cutoffs = None
    
    print("\n...")
    
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
        print("...")
        return x_tile_optimization(df, risk_scores, outcome_col, 
                                   min_group_size=0.05, min_events_per_group=2)
    
    results_df = pd.DataFrame(results)
    
    print("\n✓ :")
    print(f"  Low-risk ≤ {best_cutoffs[0]:.3f} ({best_cutoffs[0]*100:.1f}%)")
    print(f"  Intermediate-risk ≤ {best_cutoffs[1]:.3f} ({best_cutoffs[1]*100:.1f}%)")
    print(f"  High-risk > {best_cutoffs[1]:.3f}")
    print(f"  Differences in recurrence rates between high-risk and low-risk groups: {best_diff:.1f}%")
    
    return results_df, best_cutoffs


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
            -risk_scores,  
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
        print("")
        print("=" * 80)
        
        train_risk_scores = self.risk_model.predict_risk_score(train_df)
        
        results = {}
        
        print("\n" + "=" * 80)
        print("1. ")
        print("=" * 80)
        
        train_cindex = self.bootstrap_c_index(train_df, train_risk_scores)
        print(f"\nC-index: {train_cindex['c_index']:.3f} (95% CI: {train_cindex['ci_lower']:.3f}-{train_cindex['ci_upper']:.3f})")
        
        train_brier = self.calculate_brier_score(train_df, eval_times)
        print(f"\nBrier Score : {train_brier['mean']:.4f}")
        for t in eval_times:
            col = f'{t}mo'
            if col in train_brier and not np.isnan(train_brier[col]):
                print(f"  {t}mouths: {train_brier[col]:.4f}")
        
        results['train'] = {
            'cindex': train_cindex,
            'brier': train_brier,
            'risk_scores': train_risk_scores
        }
        
        if valid_df is not None:
            print("\n" + "=" * 80)
            print("2. ")
            print("=" * 80)
            
            valid_risk_scores = self.risk_model.predict_risk_score(valid_df)
            
            valid_cindex = self.bootstrap_c_index(valid_df, valid_risk_scores)
            print(f"\nC-index: {valid_cindex['c_index']:.3f} (95% CI: {valid_cindex['ci_lower']:.3f}-{valid_cindex['ci_upper']:.3f})")
            
            valid_brier = self.calculate_brier_score(valid_df, eval_times)
            print(f"\nBrier Score : {valid_brier['mean']:.4f}")
            for t in eval_times:
                col = f'{t}mo'
                if col in valid_brier and not np.isnan(valid_brier[col]):
                    print(f"  {t}mouths: {valid_brier[col]:.4f}")
            
            results['valid'] = {
                'cindex': valid_cindex,
                'brier': valid_brier,
                'risk_scores': valid_risk_scores
            }
        
        return results


def plot_kaplan_meier_single(df, groups, title, suffix, output_dir):
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.15)
    ax = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    
    group_configs = [
        ('Low-risk', '#2ecc71', 'Low-risk'),
        ('Moderate-risk', '#f39c12', 'Moderate-risk'),
        ('High-risk', '#e74c3c', 'High-risk'),
    ]
    
    kmfitters = []
    group_data = []
    group_sizes = []
    for group_name, color, label_en in group_configs:
        mask = groups == group_name
        if np.sum(mask) == 0:
            continue
        
        group_df = df.loc[mask]
        kmf = KaplanMeierFitter()
        kmf.fit(group_df['time'], event_observed=group_df['FUFA'], label=label_en)
        kmfitters.append((kmf, color, group_name))
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2, ci_show=False)
        group_data.append((kmf, group_name, color))
        group_sizes.append(len(group_df))
    
   
    max_time = df['time'].max()
    x_ticks = np.arange(0, max_time + 24, 24)
    ax.set_xticks(x_ticks)
    
    ax.set_title(f'{title} - Kaplan-Meier Survival Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Follow-up Time (Months)', fontsize=12)
    ax.set_ylabel('Recurrence-free Survival', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(False)  
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1.02)
    
 
    ax_table.axis('tight')
    ax_table.axis('off')
    
  
    table_data = []
    row_labels = []
    
    for i, (kmf, group_name, color) in enumerate(group_data):
        mask = groups == group_name
        group_df = df.loc[mask]
        total = len(group_df)
        events = group_df['FUFA'].sum()
        censored = total - events
        
      
        remaining_counts = []
        for t in x_ticks:
            try:
                surv_prob = kmf.predict(t) if t <= max_time else 0
                remaining = int(round(total * surv_prob))
                remaining = max(0, remaining)
            except:
                remaining = 0
            remaining_counts.append(remaining)
        
        table_data.append(remaining_counts)
        row_labels.append(group_name)
    
  
    if len(table_data) > 0:
        
        table = ax_table.table(cellText=table_data,
                             rowLabels=row_labels,
                             colLabels=[f'{int(t)}' for t in x_ticks],
                             loc='center',
                             cellLoc='right')  
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        
        for i in range(len(table_data)):
            for j in range(len(x_ticks)):
                table[(i+1, j)].set_facecolor('white')
        for i in range(len(row_labels)):
            table[(i+1, -1)].set_facecolor(group_configs[i][1])
            table[(i+1, -1)].set_text_props(color='white', weight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_KM_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {title} KM curve saved: 01_KM_{suffix}.png")

def plot_comparison_kaplan_meier(train_df, valid_df, train_groups, valid_groups, 
                                  best_cutoffs, output_dir):
    plot_kaplan_meier_single(train_df, train_groups, 'Training Cohort', 'train', output_dir)
    plot_kaplan_meier_single(valid_df, valid_groups, 'Validation Cohort', 'valid', output_dir)

def plot_metrics_single(results, which, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = ['C-index', 'Mean Brier Score']
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
    ax1.set_title(f'{which} Model Evaluation Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12)
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
    ax2.set_xticklabels([f'{t} months' for t in eval_times], fontsize=11)
    ax2.set_title(f'{which} Brier Score Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Brier Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for i, v in enumerate(briers):
        ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_metrics_{which}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {which} metrics plot saved: 02_metrics_{which}.png")

def plot_metrics_comparison(results, output_dir):
    plot_metrics_single(results, 'train', output_dir)
    plot_metrics_single(results, 'valid', output_dir)

def plot_risk_distribution_single(df, scores, title, suffix, best_cutoffs, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    events = df['FUFA'].values
    ax.hist(scores[events == 0], bins=30, alpha=0.6, 
            label='No recurrence', color='#2ecc71', density=False, edgecolor='black')
    ax.hist(scores[events == 1], bins=30, alpha=0.6, 
            label='Recurrence', color='#e74c3c', density=False, edgecolor='black')
    ax.axvline(best_cutoffs[0], color='black', linestyle='--', linewidth=2, 
               label=f'Low/Moderate: {best_cutoffs[0]:.3f}')
    ax.axvline(best_cutoffs[1], color='black', linestyle='--', linewidth=2, 
               label=f'Moderate/High: {best_cutoffs[1]:.3f}')
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title(f'{title} - Risk Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_yticks([25, 50, 75, 100])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_risk_distribution_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {title} risk distribution plot saved: 03_risk_distribution_{suffix}.png")

def plot_risk_distribution(train_df, valid_df, train_scores, valid_scores, 
                          best_cutoffs, output_dir):
    plot_risk_distribution_single(train_df, train_scores, 'Training Cohort', 'train', best_cutoffs, output_dir)
    plot_risk_distribution_single(valid_df, valid_scores, 'Validation Cohort', 'valid', best_cutoffs, output_dir)

def plot_risk_groups_single(df, groups, title, suffix, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    group_names = ['Low-risk', 'Moderate-risk', 'High-risk']
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
    ax1.set_title(f'{title} - Number of Patients by Risk Group', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Patients', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, 
                f'{group_stats[i]["n"]} ({group_stats[i]["n_event"]})', 
                ha='center', va='bottom', fontweight='bold')
    
    ax2 = axes[1]
    bars2 = ax2.bar([s['name'] for s in group_stats], [s['rate'] for s in group_stats], 
                   color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title(f'{title} - Recurrence Rate by Risk Group', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Recurrence Rate (%)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, 
                f'{group_stats[i]["rate"]:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_risk_groups_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {title} risk groups summary plot saved: 04_risk_groups_{suffix}.png")

def plot_risk_groups_summary(train_df, valid_df, train_groups, valid_groups, output_dir):
    plot_risk_groups_single(train_df, train_groups, 'Training Cohort', 'train', output_dir)
    plot_risk_groups_single(valid_df, valid_groups, 'Validation Cohort', 'valid', output_dir)


def calculate_hazard_ratio(df, groups, duration_col='time', event_col='FUFA'):
    """
    ）
    """
    hr_results = {}
    
    try:
        
        group_mapping = {'Low-risk': 0, 'Moderate-risk': 1, 'High-risk': 2}
        df_analysis = df.copy()
        df_analysis['risk_group_numeric'] = [group_mapping.get(g, -1) for g in groups]
        
        
        df_analysis = df_analysis[df_analysis['risk_group_numeric'] != -1]
        
        if len(df_analysis) == 0:
            return hr_results
        
        
        df_low_ref = df_analysis.copy()
        df_low_ref['is_moderate'] = (df_low_ref['risk_group_numeric'] == 1).astype(int)
        df_low_ref['is_high'] = (df_low_ref['risk_group_numeric'] == 2).astype(int)
        
        cph_low_ref = CoxPHFitter()
        cph_low_ref.fit(df_low_ref[[duration_col, event_col, 'is_moderate', 'is_high']],
                       duration_col=duration_col, event_col=event_col)
        
        if 'is_moderate' in cph_low_ref.summary.index:
            hr_results['Moderate vs Low'] = {
                'HR': float(np.exp(cph_low_ref.summary.loc['is_moderate', 'coef'])),
                'HR_lower': float(np.exp(cph_low_ref.summary.loc['is_moderate', 'coef lower 95%'])),
                'HR_upper': float(np.exp(cph_low_ref.summary.loc['is_moderate', 'coef upper 95%'])),
                'p_value': float(cph_low_ref.summary.loc['is_moderate', 'p'])
            }
        
        if 'is_high' in cph_low_ref.summary.index:
            hr_results['High vs Low'] = {
                'HR': float(np.exp(cph_low_ref.summary.loc['is_high', 'coef'])),
                'HR_lower': float(np.exp(cph_low_ref.summary.loc['is_high', 'coef lower 95%'])),
                'HR_upper': float(np.exp(cph_low_ref.summary.loc['is_high', 'coef upper 95%'])),
                'p_value': float(cph_low_ref.summary.loc['is_high', 'p'])
            }
        
       
        df_moderate_ref = df_analysis[df_analysis['risk_group_numeric'].isin([1, 2])].copy()
        if len(df_moderate_ref) >= 2: 
            df_moderate_ref['is_high'] = (df_moderate_ref['risk_group_numeric'] == 2).astype(int)
            
            cph_moderate_ref = CoxPHFitter()
            cph_moderate_ref.fit(df_moderate_ref[[duration_col, event_col, 'is_high']],
                               duration_col=duration_col, event_col=event_col)
            
            if 'is_high' in cph_moderate_ref.summary.index:
                hr_results['High vs Moderate'] = {
                    'HR': float(np.exp(cph_moderate_ref.summary.loc['is_high', 'coef'])),
                    'HR_lower': float(np.exp(cph_moderate_ref.summary.loc['is_high', 'coef lower 95%'])),
                    'HR_upper': float(np.exp(cph_moderate_ref.summary.loc['is_high', 'coef upper 95%'])),
                    'p_value': float(cph_moderate_ref.summary.loc['is_high', 'p'])
                }
            
    except Exception as e:
        print(f"HR calculation failed: {e}")
        
    return hr_results


def apply_risk_stratification(df, risk_scores, best_cutoffs):
    groups = np.where(risk_scores <= best_cutoffs[0], 'Low-risk',
             np.where(risk_scores <= best_cutoffs[1], 'Moderate-risk', 'High-risk'))
    return groups


def main_complete(train_file, valid_file, output_dir=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    if output_dir is None:
        output_dir = f'cox_validation_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Frontal Lobe Epilepsy Postoperative Recurrence Risk Stratification - Complete Validation Pipeline (with New Features)")
    print("=" * 80)
    
    feature_cols = ['onset age', 'duration', 'frequency', 'Interictal EEG', 'LYMPH', 
                   'ALT', 'MRI', 'PETSUVmax', 'AI', 'R0 Resection', 'Abnormal Post-EEG', 'Seizure in 6m']
    
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFeatures used: {', '.join(feature_cols)}")
    
    train_df, _ = load_and_preprocess_data(train_file, handle_missing='impute')
    valid_df, _ = load_and_preprocess_data(valid_file, handle_missing='impute')
    
    print("\n" + "=" * 80)
    print("Step 1: Train Cox model on training cohort")
    model = RiskPredictionModel()
    model.fit(train_df, feature_cols, duration_col='time', event_col='FUFA')
    
    print("\n" + "=" * 80)
    print("Step 2: Determine cutoffs using X-tile on training cohort")
    train_risk_scores = model.predict_risk_score(train_df)
    _, best_cutoffs = x_tile_optimization(train_df, train_risk_scores)
    
    print("\n" + "=" * 80)
    print("Step 3: Model validation")
    validator = ModelValidator(model)
    results = validator.validate(train_df, valid_df)
    
    print("\n" + "=" * 80)
    print("Step 4: Risk stratification")
    print("=" * 80)
    
    train_groups = apply_risk_stratification(train_df, train_risk_scores, best_cutoffs)
    valid_risk_scores = model.predict_risk_score(valid_df)
    valid_groups = apply_risk_stratification(valid_df, valid_risk_scores, best_cutoffs)
    
    for name, df, groups in [('Training Cohort', train_df, train_groups), 
                            ('Validation Cohort', valid_df, valid_groups)]:
        print(f"\n{name} risk stratification results:")
        print("-" * 50)
        for g in ['Low-risk', 'Moderate-risk', 'High-risk']:
            mask = groups == g
            n = np.sum(mask)
            n_event = np.sum(df.loc[mask, 'FUFA'])
            rate = n_event / n * 100 if n > 0 else 0
            print(f"  {g}: {n} patients, {n_event} events, rate {rate:.1f}%")
    
    print("\n" + "=" * 80)
    print("Step 5: Log-rank test and Hazard Ratio calculation")
    print("=" * 80)
    
    logrank_results = {}
    hr_results = {}
    
    for name, df, groups in [('Training Cohort', train_df, train_groups), 
                            ('Validation Cohort', valid_df, valid_groups)]:
        print(f"\n{name} Log-rank test:")
        logrank_results[name] = {}
        group_numeric = pd.Categorical(groups, categories=['Low-risk', 'Moderate-risk', 'High-risk']).codes
        try:
            lr_result = multivariate_logrank_test(df['time'], group_numeric, df['FUFA'])
            print(f"  All groups: χ² = {lr_result.test_statistic:.2f}, p = {lr_result.p_value:.6f}")
            logrank_results[name]['All groups'] = {'chi_sq': lr_result.test_statistic, 'p_value': lr_result.p_value}
            
            group_pairs = [('Low-risk', 'Moderate-risk'), ('Low-risk', 'High-risk'), ('Moderate-risk', 'High-risk')]
            for g1, g2 in group_pairs:
                mask1 = groups == g1
                mask2 = groups == g2
                lr_pair = logrank_test(df.loc[mask1, 'time'], df.loc[mask2, 'time'],
                                      df.loc[mask1, 'FUFA'], df.loc[mask2, 'FUFA'])
                sig = '***' if lr_pair.p_value < 0.001 else '**' if lr_pair.p_value < 0.01 else '*' if lr_pair.p_value < 0.05 else ''
                print(f"  {g1} vs {g2}: χ² = {lr_pair.test_statistic:.2f}, p = {lr_pair.p_value:.4f} {sig}")
                logrank_results[name][f"{g1} vs {g2}"] = {'chi_sq': lr_pair.test_statistic, 'p_value': lr_pair.p_value}
        except Exception as e:
            print(f"  Calculation failed: {e}")
        
        
        print(f"\n{name} Hazard Ratio:")
        hr = calculate_hazard_ratio(df, groups)
        hr_results[name] = hr
        for comparison, hr_info in hr.items():
            sig = '***' if hr_info['p_value'] < 0.001 else '**' if hr_info['p_value'] < 0.01 else '*' if hr_info['p_value'] < 0.05 else ''
            print(f"  {comparison}: HR = {hr_info['HR']:.2f} (95% CI: {hr_info['HR_lower']:.2f}-{hr_info['HR_upper']:.2f}), p = {hr_info['p_value']:.4f} {sig}")
    
    print("\n" + "=" * 80)
    print("Step 6: Generate visualization results")
    print("=" * 80)
    
    plot_comparison_kaplan_meier(train_df, valid_df, train_groups, valid_groups, 
                                  best_cutoffs, output_dir)
    plot_metrics_comparison(results, output_dir)
    plot_risk_distribution(train_df, valid_df, train_risk_scores, valid_risk_scores, 
                          best_cutoffs, output_dir)
    plot_risk_groups_summary(train_df, valid_df, train_groups, valid_groups, output_dir)
    
    print("\n" + "=" * 80)
    print("Step 7: Save stratified results")
    print("=" * 80)
    
    train_df['risk_score'] = train_risk_scores
    train_df['risk_group'] = train_groups
    train_df.to_csv(f'{output_dir}/training_set_with_risk.csv', index=False, encoding='utf-8-sig')
    print(f"✓ Training cohort stratified results saved: training_set_with_risk.csv")
    
    valid_df['risk_score'] = valid_risk_scores
    valid_df['risk_group'] = valid_groups
    valid_df.to_csv(f'{output_dir}/validation_set_with_risk.csv', index=False, encoding='utf-8-sig')
    print(f"✓ Validation cohort stratified results saved: validation_set_with_risk.csv")
    
    
    summary_metrics = []
    
    
    summary_metrics.append({'Metric': 'C-index (Training)', 'Value': f"{results['train']['cindex']['c_index']:.3f}"})
    summary_metrics.append({'Metric': 'C-index (Validation)', 'Value': f"{results['valid']['cindex']['c_index']:.3f}" if 'valid' in results else 'N/A'})
    summary_metrics.append({'Metric': 'C-index 95% CI (Training)', 'Value': f"{results['train']['cindex']['ci_lower']:.3f}-{results['train']['cindex']['ci_upper']:.3f}"})
    summary_metrics.append({'Metric': 'C-index 95% CI (Validation)', 'Value': f"{results['valid']['cindex']['ci_lower']:.3f}-{results['valid']['cindex']['ci_upper']:.3f}" if 'valid' in results else 'N/A'})
    summary_metrics.append({'Metric': 'Mean Brier Score (Training)', 'Value': f"{results['train']['brier']['mean']:.4f}"})
    summary_metrics.append({'Metric': 'Mean Brier Score (Validation)', 'Value': f"{results['valid']['brier']['mean']:.4f}" if 'valid' in results else 'N/A'})
    summary_metrics.append({'Metric': 'Low-risk cutoff', 'Value': f"{best_cutoffs[0]:.3f}"})
    summary_metrics.append({'Metric': 'Moderate-risk cutoff', 'Value': f"{best_cutoffs[1]:.3f}"})
    
    
    summary_metrics.append({'Metric': '--- Log-rank test ---', 'Value': ''})
    
    
    if 'Training Cohort' in logrank_results:
        for comparison, stats in logrank_results['Training Cohort'].items():
            summary_metrics.append({
                'Metric': f"Log-rank {comparison} (Training)",
                'Value': f"χ²={stats['chi_sq']:.2f}, p={stats['p_value']:.4f}"
            })
    
    
    if 'Validation Cohort' in logrank_results:
        for comparison, stats in logrank_results['Validation Cohort'].items():
            summary_metrics.append({
                'Metric': f"Log-rank {comparison} (Validation)",
                'Value': f"χ²={stats['chi_sq']:.2f}, p={stats['p_value']:.4f}"
            })
    
   
    summary_metrics.append({'Metric': '--- Hazard Ratio ---', 'Value': ''})
    
   
    if 'Training Cohort' in hr_results:
        for comparison, hr_info in hr_results['Training Cohort'].items():
            summary_metrics.append({
                'Metric': f"HR {comparison} (Training)",
                'Value': f"HR={hr_info['HR']:.2f} (95%CI: {hr_info['HR_lower']:.2f}-{hr_info['HR_upper']:.2f}), p={hr_info['p_value']:.4f}"
            })
    
    if 'Validation Cohort' in hr_results:
        for comparison, hr_info in hr_results['Validation Cohort'].items():
            summary_metrics.append({
                'Metric': f"HR {comparison} (Validation)",
                'Value': f"HR={hr_info['HR']:.2f} (95%CI: {hr_info['HR_lower']:.2f}-{hr_info['HR_upper']:.2f}), p={hr_info['p_value']:.4f}"
            })
    
    pd.DataFrame(summary_metrics).to_csv(f'{output_dir}/summary_results.csv', index=False, encoding='utf-8-sig')
    print("✓ Summary results saved: summary_results.csv")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return {
        'model': model,
        'best_cutoffs': best_cutoffs,
        'results': results,
        'train_df': train_df,
        'valid_df': valid_df,
        'train_groups': train_groups,
        'valid_groups': valid_groups,
        'hr_results': hr_results,
        'logrank_results': logrank_results
    }

if __name__ == "__main__":

    result = main_complete(
        train_file='5C_co5.csv',
        valid_file='5C_co6.csv',
        output_dir='cox_validation_complete_new_features'
    )
