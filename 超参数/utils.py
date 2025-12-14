import torch
import torch.nn as nn
import torch.nn.functional as F  # 补充缺失的引用
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # 移除 seaborn 避免报错
import os

class MetricTracker:
    def __init__(self, threshold=0.1, time_tolerance=1.0):
        self.threshold = threshold
        self.time_tolerance = time_tolerance
        self.reset()

    def reset(self):
        self.preds_time = []
        self.targets_time = []
        self.preds_val = []
        self.targets_val = []
        
        # Stats for Rate metrics
        self.A = 0 # Pred Pos (Predicted as Attack)
        self.B = 0 # True Pos (Actually Attack)
        self.C = 0 # Hits

    def update(self, pred_t, pred_v, target_t, target_v):
        """
        pred_t: (B, 1)
        pred_v: (B, 2)
        target_t: (B, 1)
        target_v: (B, 2)
        """
        # Detach and CPU
        pt = pred_t.detach().cpu().numpy().flatten()
        tt = target_t.detach().cpu().numpy().flatten()
        pv = pred_v.detach().cpu().numpy() # (B, 2)
        tv = target_v.detach().cpu().numpy() # (B, 2)
        
        self.preds_time.extend(pt)
        self.targets_time.extend(tt)
        self.preds_val.extend(pv)
        self.targets_val.extend(tv)
        
        # Hit Rate Logic
        # "Has Attack": Value > 0 (or threshold). Since data is [0,1], use small threshold.
        # Check based on 'count' (index 0 of val)
        pred_has_attack = pv[:, 0] > self.threshold
        true_has_attack = tv[:, 0] > self.threshold
        
        self.A += np.sum(pred_has_attack)
        self.B += np.sum(true_has_attack)
        
        # Hit: Pred is Positive AND True is Positive AND Time Error <= Tolerance
        time_diff = np.abs(pt - tt)
        time_match = time_diff <= self.time_tolerance
        
        hits = pred_has_attack & true_has_attack & time_match
        self.C += np.sum(hits)

    def calculate_metrics(self):
        pt = np.array(self.preds_time)
        tt = np.array(self.targets_time)
        pv = np.array(self.preds_val)
        tv = np.array(self.targets_val)
        
        # 1. MAE, RMSE, SMAPE for Values (Count)
        # Using Count (idx 0) for main metric
        p_count = pv[:, 0]
        t_count = tv[:, 0]
        
        mae = np.mean(np.abs(p_count - t_count))
        rmse = np.sqrt(np.mean((p_count - t_count)**2))
        
        # SMAPE
        denom = (np.abs(p_count) + np.abs(t_count)) / 2.0
        smape = np.mean(np.abs(p_count - t_count) / (denom + 1e-8))
        
        # Peak-MAE: Top 20% high intensity samples
        threshold_top20 = np.percentile(t_count, 80)
        peak_indices = t_count >= threshold_top20
        if np.sum(peak_indices) > 0:
            peak_mae = np.mean(np.abs(p_count[peak_indices] - t_count[peak_indices]))
        else:
            peak_mae = 0.0
            
        # Rates
        time_hit_rate = self.C / (self.A + 1e-8)
        miss_rate = 1 - (self.C / (self.B + 1e-8))
        total_hit = self.C 
        total_miss = self.B - self.C # Actually missed positives
        
        return {
            "MAE": mae,
            "RMSE": rmse,
            "SMAPE": smape,
            "Peak-MAE": peak_mae,
            "Time_Hit_Rate": time_hit_rate,
            "Miss_Rate": miss_rate,
            "Total_Hit": total_hit,
            "Total_Miss": total_miss
        }

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
        
    def forward(self, pred, target):
        return F.huber_loss(pred, target, delta=self.delta)

def visualize_results(history, save_path="results"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Plot Loss
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Training Curve")
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

def visualize_ablation(results_df, save_path="results"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Simple bar chart for MAE across configs using pure Matplotlib
    plt.figure(figsize=(10, 6))
    
    # Check if DataFrame is not empty
    if len(results_df) > 0:
        configs = results_df['Config']
        mae_values = results_df['MAE']
        
        plt.bar(configs, mae_values, color='skyblue', edgecolor='black', alpha=0.7)
        
        plt.xlabel("Configuration")
        plt.ylabel("MAE")
        plt.title("Ablation Study: MAE by Configuration")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "ablation_mae.png"))
    plt.close()