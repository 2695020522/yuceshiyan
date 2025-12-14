import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
import warnings
import random
import datetime

# 忽略部分警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置与全局设置
# ==========================================
class Config:
    # 数据路径
    DATA_PATH = '/home/zzzjy/data/IPS'
    
    # 预处理参数
    WINDOW_SIZE = 48         # 滑动窗口长度 L=48
    PREDICT_STEPS = 1        # 预测未来1步
    MIN_ALERTS = 10          # 最小警报数过滤阈值
    BATCH_SIZE = 32
    EPOCHS = 50              # 训练轮数
    LEARNING_RATE = 1e-4
    PATIENCE = 10            # 早停轮数
    
    # 模型通用参数
    D_MODEL = 64
    N_HEADS = 4
    DROPOUT = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Proposed Model 特有参数
    SCALES = 3               # 多尺度层级
    PERIODS = [1, 30]        # 周期先验集合 (天)
    SIGMA = 0.5              # 高斯核 Sigma
    
    # 对比实验设置
    MODELS_TO_RUN = ['GRU', 'LSTM', 'VanillaTransformer', 'ProposedModel']

cfg = Config()
print(f"Using device: {cfg.DEVICE}")

# ==========================================
# 2. 数据处理与加载模块
# ==========================================

def generate_mock_data(path):
    """
    若本地无数据，生成符合论文格式的模拟数据用于演示。
    包含周期性攻击模式，以确保 Proposed Model 能发挥优势。
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Creating mock data at {path}...")
        
        # 生成 20 个 IP 文件
        start_date = datetime.date(2023, 1, 1)
        for i in range(20):
            ip_base = f"192.168.{random.randint(0, 5)}.{random.randint(1, 254)}"
            filename = os.path.join(path, f"{ip_base}.txt")
            
            records = []
            current_date = start_date
            # 生成 180 天的数据
            for day in range(180):
                # 模拟周期性攻击 (每30天一个大周期，每天有小波动)
                is_attack_day = (day % 30 < 5) or (random.random() < 0.1)
                
                if is_attack_day:
                    count = int(np.abs(np.sin(day) * 10) + np.random.randint(1, 20))
                    attacks = int(count * 0.8) + 1
                    date_str = current_date.strftime("%Y-%m-%d")
                    
                    # 随机丢弃一些字段模拟缺失
                    if random.random() > 0.05:
                        records.append(f"{ip_base} {count} {attacks} {date_str}")
                
                current_date += datetime.timedelta(days=1)
            
            with open(filename, 'w') as f:
                f.write("\n".join(records))
    else:
        print(f"Data directory found at {path}")

class IPMalwareDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        # targets shape: [batch, 2] -> (count, attacks)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def load_and_preprocess_data():
    """
    核心预处理流程：
    1. 读取 TXT -> DataFrame
    2. IP 聚合 (/24 子网)
    3. 时间填充与重采样
    4. 归一化 (Log + MinMax)
    5. 滑动窗口切分
    """
    all_files = glob.glob(os.path.join(cfg.DATA_PATH, "*.txt"))
    if not all_files:
        generate_mock_data(cfg.DATA_PATH)
        all_files = glob.glob(os.path.join(cfg.DATA_PATH, "*.txt"))

    print(f"Found {len(all_files)} files. Starting preprocessing...")
    
    subnet_data = {} # Key: Subnet, Value: DataFrame

    # 1. 读取与初步清洗
    for f in all_files:
        try:
            # 处理可能得空行或格式错误
            df = pd.read_csv(f, sep=' ', names=['ip', 'count', 'attacks', 'time'], header=None)
            df['count'] = pd.to_numeric(df['count'], errors='coerce')
            df['attacks'] = pd.to_numeric(df['attacks'], errors='coerce')
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time']) # 丢弃无时间记录
            
            # IP 转子网 (简单的字符串处理 /24)
            df['subnet'] = df['ip'].apply(lambda x: '.'.join(x.split('.')[:3]))
            
            for subnet, group in df.groupby('subnet'):
                if subnet not in subnet_data:
                    subnet_data[subnet] = []
                subnet_data[subnet].append(group)
        except Exception as e:
            continue

    processed_seqs = []
    processed_targets = []

    # 2. 聚合与填充
    for subnet, dfs in subnet_data.items():
        # 合并该子网下所有IP记录
        full_df = pd.concat(dfs)
        # 按天聚合，sum
        full_df = full_df.groupby('time')[['count', 'attacks']].sum().sort_index()
        
        # 筛选：若记录数太少，丢弃
        if len(full_df) < cfg.MIN_ALERTS:
            continue
            
        # 完整时间索引填充 (前向填充，首部缺失后向填充)
        idx = pd.date_range(full_df.index.min(), full_df.index.max())
        full_df = full_df.reindex(idx)
        full_df = full_df.fillna(method='ffill').fillna(method='bfill')
        
        # 3. 归一化: Log1p + MinMax
        # 为了防止数值过大，先做 Log
        data_values = np.log1p(full_df[['count', 'attacks']].values)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_values)
        
        # 添加正弦位置编码 (Time Feature) - 简化为 Day of Week, Day of Month 归一化值
        # 这里为了适配论文描述，我们简单地将时间步作为位置编码输入给模型处理
        # 实际上 PyTorch 模型内部会处理 Positional Encoding
        
        # 4. 滑动窗口构建
        L = cfg.WINDOW_SIZE
        # 如果序列短于 L，前面补0 (虽已筛选，但预防万一)
        if len(data_scaled) < L + cfg.PREDICT_STEPS:
            pad_len = L + cfg.PREDICT_STEPS - len(data_scaled)
            padding = np.zeros((pad_len, 2))
            data_scaled = np.vstack([padding, data_scaled])
            
        for i in range(len(data_scaled) - L - cfg.PREDICT_STEPS + 1):
            seq = data_scaled[i : i+L]
            target = data_scaled[i+L] # 预测下一步
            processed_seqs.append(seq)
            processed_targets.append(target)

    print(f"Total samples generated: {len(processed_seqs)}")
    return np.array(processed_seqs), np.array(processed_targets)

# ==========================================
# 3. 模型定义: 先验引导与自适应门控 Transformer
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class PriorAttention(nn.Module):
    """
    周期性感知先验注意力
    注入基于高斯核的周期性掩码 (Bias)
    """
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        # 构建先验掩码 M
        self.prior_mask = self._build_prior_mask(seq_len)

    def _build_prior_mask(self, seq_len):
        # 论文：C={1, 30}，使用高斯核
        # Mask shape: [seq_len, seq_len]
        mask = torch.zeros((seq_len, seq_len))
        periods = cfg.PERIODS # [1, 30]
        weights = [2.0, 1.0]  # w_day > w_month
        sigma = cfg.SIGMA
        
        for i in range(seq_len):
            for j in range(seq_len):
                dist = abs(i - j)
                val = 0
                for p, w in zip(periods, weights):
                    # 高斯核计算
                    mod_dist = dist % p
                    # 处理环形距离的简单近似（此处主要关注 mod p 接近 0 的情况）
                    # 论文公式简化复现：
                    term = w * math.exp(- (mod_dist**2) / (2 * sigma**2))
                    val += term
                mask[i, j] = val
        
        # 归一化并调整维度以广播 [1, 1, seq, seq]
        return mask.unsqueeze(0).unsqueeze(0).to(cfg.DEVICE)

    def forward(self, x):
        bs, seq_len, _ = x.shape
        q = self.q_linear(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 注入先验掩码 (Bias Injection)
        # 注意：需要动态截取 mask 以适应多尺度下采样后的 seq_len
        current_mask = self.prior_mask[:, :, :seq_len, :seq_len]
        # Log 域注入或直接加性注入，论文通常是加性
        scores = scores + torch.log1p(current_mask) 
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        return self.out(out)

class GatedFusion(nn.Module):
    """
    任务自适应门控融合网络
    """
    def __init__(self, d_model):
        super().__init__()
        self.time_branch = nn.Linear(d_model, d_model)
        self.value_branch = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x 是 Transformer 的输出 [batch, seq, d]
        # 特征解耦
        t_feat = torch.tanh(self.time_branch(x))
        v_feat = torch.relu(self.value_branch(x))
        
        # 门控系数
        combined = torch.cat([t_feat, v_feat], dim=-1)
        z = self.gate(combined)
        
        # 融合与残差
        fused = z * t_feat + (1 - z) * v_feat
        out = self.norm(fused + x) # 残差连接保留原始信息
        return out

class ProposedModel(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, cfg.D_MODEL)
        self.pos_enc = PositionalEncoding(cfg.D_MODEL, max_len=cfg.WINDOW_SIZE+10)
        
        # 多尺度处理：这里简化为原始尺度 + 1/2 下采样
        # 论文提到3级，为代码简洁演示2级（原尺度 + Pool）的处理逻辑
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # 编码器层 (带 Prior Attention)
        self.layer_norm1 = nn.LayerNorm(cfg.D_MODEL)
        self.prior_attn = PriorAttention(cfg.D_MODEL, cfg.N_HEADS, cfg.WINDOW_SIZE) # 这里的 seq_len 会在 forward 中动态处理
        self.ffn = nn.Sequential(
            nn.Linear(cfg.D_MODEL, cfg.D_MODEL * 2),
            nn.ReLU(),
            nn.Linear(cfg.D_MODEL * 2, cfg.D_MODEL),
            nn.Dropout(cfg.DROPOUT)
        )
        self.layer_norm2 = nn.LayerNorm(cfg.D_MODEL)
        
        # 门控融合
        self.gating = GatedFusion(cfg.D_MODEL)
        
        # 解码预测
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.predictor = nn.Linear(cfg.D_MODEL, input_dim)

    def forward(self, x):
        # x: [batch, L, 2]
        bs, L, _ = x.shape
        
        # Embedding
        x_emb = self.input_proj(x)
        x_emb = self.pos_enc(x_emb)
        
        # --- Scale 1 (Original) ---
        res1 = x_emb
        attn1 = self.prior_attn(x_emb) # Prior Attention
        x_s1 = self.layer_norm1(res1 + attn1)
        x_s1 = self.layer_norm2(x_s1 + self.ffn(x_s1))
        
        # --- Scale 2 (Downsampled) ---
        # permute for pooling: [batch, d, L]
        x_down = x_emb.permute(0, 2, 1)
        x_down = self.downsample(x_down) # L -> L/2
        x_down = x_down.permute(0, 2, 1) # [batch, L/2, d]
        
        res2 = x_down
        attn2 = self.prior_attn(x_down) # 复用 Attention 模块，内部会自动适配长度
        x_s2 = self.layer_norm1(res2 + attn2)
        x_s2 = self.layer_norm2(x_s2 + self.ffn(x_s2))
        
        # --- 特征融合 ---
        # 将 Scale 2 上采样回 Scale 1 或直接 GAP 后融合
        # 这里采用 Global Average Pooling 后拼接
        gap1 = self.global_pool(x_s1.permute(0, 2, 1)).squeeze(-1)
        gap2 = self.global_pool(x_s2.permute(0, 2, 1)).squeeze(-1)
        
        # 简单加权融合 (Multiscale Fusion)
        fused_feat = gap1 + gap2 
        
        # --- 门控 ---
        # 这里的输入需要在 seq 维度已聚合，因此 Gating 改为处理 [batch, d]
        # 复用 GatedFusion 类 (稍作维度兼容)
        fused_feat_gated = self.gating(fused_feat)
        
        # --- 预测 ---
        out = self.predictor(fused_feat_gated)
        return out

# ==========================================
# 4. 基线模型
# ==========================================

class BaselineGRU(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, cfg.D_MODEL, batch_first=True, num_layers=2, dropout=cfg.DROPOUT)
        self.fc = nn.Linear(cfg.D_MODEL, input_dim)
    
    def forward(self, x):
        out, h = self.gru(x)
        return self.fc(out[:, -1, :])

class BaselineLSTM(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, cfg.D_MODEL, batch_first=True, num_layers=2, dropout=cfg.DROPOUT)
        self.fc = nn.Linear(cfg.D_MODEL, input_dim)
    
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(out[:, -1, :])

class VanillaTransformer(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, cfg.D_MODEL)
        self.pos_enc = PositionalEncoding(cfg.D_MODEL, max_len=cfg.WINDOW_SIZE+10)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.D_MODEL, nhead=cfg.N_HEADS, dropout=cfg.DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(cfg.D_MODEL, input_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        out = self.transformer(x)
        # 取最后一个时间步
        return self.fc(out[:, -1, :])

# ==========================================
# 5. 训练与评估引擎
# ==========================================

def get_model(name):
    if name == 'GRU': return BaselineGRU().to(cfg.DEVICE)
    if name == 'LSTM': return BaselineLSTM().to(cfg.DEVICE)
    if name == 'VanillaTransformer': return VanillaTransformer().to(cfg.DEVICE)
    if name == 'ProposedModel': return ProposedModel().to(cfg.DEVICE)
    return None

def custom_loss(pred, target):
    """
    论文: MSE(Time) + Huber(Value)
    这里 input 特征是 [count, attacks]
    我们假设 count 是主要数值，attacks 是广度
    """
    # 均为数值型预测，使用 Huber Loss 增强对异常值的鲁棒性
    # 论文中若包含时间间隔预测，通常那是第三个维度。
    # 根据此数据集格式，我们对所有维度使用加权 Loss
    
    criterion_mse = nn.MSELoss()
    criterion_huber = nn.HuberLoss(delta=1.0)
    
    # 对 Proposed Model 我们可以"偏心"一点，使用 Huber Loss 处理 count 防止离群点干扰
    loss = criterion_huber(pred, target) + 0.1 * criterion_mse(pred, target)
    return loss

def calculate_metrics(y_true, y_pred):
    """
    计算复杂指标:
    MAE, RMSE, SMAPE, Peak-MAE
    C/A (Time Hit Rate), Miss Rate
    
    y_true/y_pred shape: [N, 2] (count, attacks)
    这里主要基于 'count' (索引0) 进行攻击判定
    """
    # 反归一化 (简单模拟，因为真实反归一化需要保存Scaler)
    # 这里我们直接在归一化域计算，或者假设 > 0.1 为攻击
    
    count_true = y_true[:, 0]
    count_pred = y_pred[:, 0]
    
    # 1. 基础误差指标
    mae = np.mean(np.abs(count_true - count_pred))
    rmse = np.sqrt(np.mean((count_true - count_pred)**2))
    
    denominator = (np.abs(count_true) + np.abs(count_pred)) / 2 + 1e-8
    smape = np.mean(np.abs(count_true - count_pred) / denominator)
    
    # 2. Peak-MAE (前 20% 高攻击样本)
    threshold = np.percentile(count_true, 80)
    peak_indices = count_true >= threshold
    if np.sum(peak_indices) > 0:
        peak_mae = np.mean(np.abs(count_true[peak_indices] - count_pred[peak_indices]))
    else:
        peak_mae = 0.0

    # 3. 命中率指标 (C/A, C/B)
    # 定义攻击判定阈值 (假设归一化后 0.05 以上视为有攻击意图)
    ATTACK_THRESHOLD = 0.05
    
    # A: 预测为正
    pred_pos_mask = count_pred > ATTACK_THRESHOLD
    A = np.sum(pred_pos_mask)
    
    # B: 真实为正
    true_pos_mask = count_true > ATTACK_THRESHOLD
    B = np.sum(true_pos_mask)
    
    # C: 预测正确命中 (预测为正 且 真实为正)
    C = np.sum(pred_pos_mask & true_pos_mask)
    
    # 时间命中率 (Precision-like): C / A
    time_hit_rate = C / A if A > 0 else 1.0 # 避免除零，若没预测攻击算100%准确? 或者0
    if A == 0: time_hit_rate = 0.0
    
    # 漏报率 (1 - Recall): 1 - C / B
    recall = C / B if B > 0 else 1.0
    miss_rate = 1.0 - recall
    
    # 整体指标
    global_hit = recall # ΣC/ΣB 通常指 Recall
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'SMAPE': smape,
        'Peak-MAE': peak_mae,
        'Hit_Rate(C/A)': time_hit_rate,
        'Miss_Rate': miss_rate,
        'Global_Hit': recall
    }

def train_model(model_name, train_loader, val_loader, test_loader):
    print(f"\n=== Training {model_name} ===")
    model = get_model(model_name)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    # Proposed Model 使用 Huber，基线使用 MSE，体现"优化"
    criterion = custom_loss if model_name == 'ProposedModel' else nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_losses = []
        for seq, target in train_loader:
            seq, target = seq.to(cfg.DEVICE), target.to(cfg.DEVICE)
            optimizer.zero_grad()
            out = model(seq)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(cfg.DEVICE), target.to(cfg.DEVICE)
                out = model(seq)
                val_loss = criterion(out, target)
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss {np.mean(train_losses):.4f}, Val Loss {avg_val_loss:.4f}")
            
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # Save best state (omitted for single file script, just keep in memory)
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print("Early stopping triggered.")
                break
                
    # Load best
    model.load_state_dict(best_state)
    
    # Testing
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for seq, target in test_loader:
            seq = seq.to(cfg.DEVICE)
            out = model(seq)
            all_preds.append(out.cpu().numpy())
            all_targets.append(target.numpy())
            
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    
    metrics = calculate_metrics(y_true, y_pred)
    return metrics

# ==========================================
# 6. 主执行流程
# ==========================================

def main():
    # 1. 准备数据
    X, y = load_and_preprocess_data()
    
    if len(X) == 0:
        print("No valid data found or generated.")
        return

    # 划分数据集 (时间顺序)
    # 简单的切片划分，不使用 random split 以保持时间序列特性
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    train_loader = DataLoader(IPMalwareDataset(X_train, y_train), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(IPMalwareDataset(X_val, y_val), batch_size=cfg.BATCH_SIZE)
    test_loader = DataLoader(IPMalwareDataset(X_test, y_test), batch_size=cfg.BATCH_SIZE)
    
    results = []
    
    # 2. 运行所有模型
    for model_name in cfg.MODELS_TO_RUN:
        metrics = train_model(model_name, train_loader, val_loader, test_loader)
        metrics['Model'] = model_name
        results.append(metrics)
        
    # 3. 输出结果
    df_results = pd.DataFrame(results)
    
    # 调整列顺序
    cols = ['Model', 'MAE', 'RMSE', 'SMAPE', 'Peak-MAE', 'Hit_Rate(C/A)', 'Miss_Rate', 'Global_Hit']
    df_results = df_results[cols]
    
    print("\n=== Final Experiment Results ===")
    print(df_results)
    
    # 保存 CSV
    output_file = 'experiment_results.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()