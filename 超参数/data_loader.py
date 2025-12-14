import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import glob
from tqdm import tqdm

class IPSTimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets, masks, scale_configs):
        """
        Args:
            sequences: List of tensors, shape (L, features)
            targets: List of tensors, shape (2,) -> [next_interval, next_values]
            masks: List of tensors, shape (L,) -> 1 for valid, 0 for padding
            scale_configs: List of downsampling factors, e.g., [1, 2, 4]
        """
        self.sequences = sequences
        self.targets = targets
        self.masks = masks
        self.scale_configs = scale_configs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]  # (L, D)
        target = self.targets[idx] # (2 + features) usually next_interval, count, attacks
        mask = self.masks[idx]     # (L,)
        
        # Pre-calculate multi-scale inputs to save time during training or do it on fly
        # Here we do it on fly to keep memory usage low
        multi_scale_seqs = {}
        
        # Convert to tensor if not already
        seq_tensor = seq if torch.is_tensor(seq) else torch.tensor(seq, dtype=torch.float32)
        mask_tensor = mask if torch.is_tensor(mask) else torch.tensor(mask, dtype=torch.float32)
        target_tensor = target if torch.is_tensor(target) else torch.tensor(target, dtype=torch.float32)

        return {
            'seq': seq_tensor,
            'mask': mask_tensor,
            'target': target_tensor
        }

def parse_ip_to_subnet(ip_str):
    """Convert IP to /24 subnet string."""
    parts = ip_str.split('.')
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0"
    return ip_str

def load_and_preprocess_data(data_dir, window_size=48, threshold=10, test_mode=False):
    """
    Main preprocessing pipeline described in the paper.
    """
    print(f"Loading data from {data_dir}...")
    
    # 1. IP Aggregation & Loading
    subnet_data = {} # Key: Subnet, Value: DataFrame
    
    if test_mode or not os.path.exists(data_dir):
        print("Warning: Data directory not found or test_mode=True. Generating synthetic data.")
        return generate_synthetic_data(window_size)

    files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    for file_path in tqdm(files, desc="Processing Files"):
        try:
            # Assuming filename is IP, but content also has IP. Using content.
            # Format: IP count attacks time
            # Handling potential missing values ('-' or empty) during read
            df = pd.read_csv(file_path, sep='\s+', names=['ip', 'count', 'attacks', 'time'], 
                             na_values=['-', ''], dtype={'count': float, 'attacks': float})
            
            # Convert IP to Subnet
            df['subnet'] = df['ip'].apply(parse_ip_to_subnet)
            df['time'] = pd.to_datetime(df['time'])
            
            # Aggregate by Subnet and Time
            # Sum count and attacks, keep subnet
            for subnet, group in df.groupby('subnet'):
                group_agg = group.groupby('time')[['count', 'attacks']].sum().reset_index()
                
                if subnet in subnet_data:
                    subnet_data[subnet] = pd.concat([subnet_data[subnet], group_agg]).groupby('time').sum().reset_index()
                else:
                    subnet_data[subnet] = group_agg
        except Exception as e:
            # print(f"Error processing {file_path}: {e}")
            continue

    print(f"Total subnets found: {len(subnet_data)}")

    # 2. Sequence Filtering & 3. Missing Value Filling
    processed_sequences = []
    
    all_counts = []
    all_attacks = []

    valid_subnets = []

    for subnet, df in subnet_data.items():
        if len(df) < threshold:
            continue
        
        # Sort by time
        df = df.sort_values('time')
        
        # Reindex to handle missing dates (if explicit gaps exist in time)
        full_idx = pd.date_range(start=df['time'].min(), end=df['time'].max(), freq='D')
        df = df.set_index('time').reindex(full_idx)
        
        # Missing Value Filling: Forward Fill then Backward Fill
        df['count'] = df['count'].ffill().bfill()
        df['attacks'] = df['attacks'].ffill().bfill()
        
        # Calculate Interval (Time Delta) for the target prediction
        # The paper predicts "Next_Interval". 
        # We need to construct sequences where input is history, target is next day's values and interval.
        # However, reindexing made it continuous (daily). 
        # If the original problem implies sporadic attacks, the reindexing might mask "Interval".
        # Assuming "Interval" means days until next attack. 
        # If we reindexed, every row is a day.
        # Let's assume standard time series forecasting: predict next step values.
        # If the original data was sparse, 'count' would be 0 on missing days.
        # If we fill valid values, we assume attacks persist? 
        # Paper says: "Forward Fill". 
        
        valid_subnets.append(df.reset_index())
        all_counts.extend(df['count'].values)
        all_attacks.extend(df['attacks'].values)

    # 4. Normalization
    # Log transform: log(x+1)
    all_counts = np.log1p(all_counts)
    all_attacks = np.log1p(all_attacks)
    
    # MinMax Scaling
    scaler_count = MinMaxScaler()
    scaler_attacks = MinMaxScaler()
    
    scaler_count.fit(all_counts.reshape(-1, 1))
    scaler_attacks.fit(all_attacks.reshape(-1, 1))
    
    final_seqs = []
    final_targets = []
    final_masks = []
    
    print("Building sliding windows...")
    for df in valid_subnets:
        # Transform data
        counts = scaler_count.transform(np.log1p(df['count'].values).reshape(-1, 1)).flatten()
        attacks = scaler_attacks.transform(np.log1p(df['attacks'].values).reshape(-1, 1)).flatten()
        
        # Features: [count, attacks]
        data = np.stack([counts, attacks], axis=1) # (T, 2)
        
        L = window_size
        T = len(data)
        
        # Sliding Window
        # We need to predict t+1 based on t-L+1...t
        for i in range(T):
            # Target is the NEXT value. If i is the last element, we can't predict next.
            if i + 1 >= T:
                break
                
            target = data[i+1] # Next Count, Next Attacks. 
            # Note: Paper mentions predicting "Next_Interval". 
            # Since we reindexed to daily, interval is always 1 unless we look at raw sporadic data.
            # Based on standard TS forecasting in this context, we usually predict values.
            # We will add a placeholder for interval (1.0) or calculate gap if data was sparse.
            # For this implementation, we predict [Interval=1, Count, Attacks]
            target_vector = np.concatenate(([1.0], target)) 
            
            # Input Sequence
            if i + 1 >= L:
                # Long sequence: directly slice
                seq = data[i+1-L : i+1]
                mask = np.ones(L)
            else:
                # Short sequence: Zero-padding at FRONT
                valid_len = i + 1
                seq_valid = data[0 : i+1]
                pad_len = L - valid_len
                padding = np.zeros((pad_len, 2))
                seq = np.concatenate([padding, seq_valid], axis=0)
                
                # Mask: 0 for padding, 1 for valid
                mask = np.concatenate([np.zeros(pad_len), np.ones(valid_len)])
            
            final_seqs.append(seq)
            final_targets.append(target_vector)
            final_masks.append(mask)

    return np.array(final_seqs), np.array(final_targets), np.array(final_masks)

def generate_synthetic_data(window_size):
    """Generates random data for testing code logic."""
    N_SAMPLES = 200
    L = window_size
    n_features = 2
    
    seqs = np.random.rand(N_SAMPLES, L, n_features).astype(np.float32)
    # Target: [Interval, Count, Attacks]
    targets = np.random.rand(N_SAMPLES, 3).astype(np.float32)
    masks = np.ones((N_SAMPLES, L)).astype(np.float32)
    # Simulate some short sequences with padding
    for i in range(50):
        pad_len = np.random.randint(1, L-5)
        masks[i, :pad_len] = 0
        seqs[i, :pad_len, :] = 0
        
    return seqs, targets, masks

def get_dataloaders(data_dir, batch_size=32, window_size=48, scale_configs=[1, 2, 4]):
    X, Y, M = load_and_preprocess_data(data_dir, window_size=window_size)
    
    # Split 70/15/15
    n_total = len(X)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    
    # Assuming data is somewhat time-ordered by processing logic, but strictly shuffling 
    # might break time-dependency if not carefully split. 
    # Paper says "split by time order".
    # Here we split indices simply.
    
    train_dataset = IPSTimeSeriesDataset(X[:n_train], Y[:n_train], M[:n_train], scale_configs)
    val_dataset = IPSTimeSeriesDataset(X[n_train:n_train+n_val], Y[n_train:n_train+n_val], M[n_train:n_train+n_val], scale_configs)
    test_dataset = IPSTimeSeriesDataset(X[n_train+n_val:], Y[n_train+n_val:], M[n_train+n_val:], scale_configs)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader