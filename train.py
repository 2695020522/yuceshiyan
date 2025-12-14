import torch
import torch.optim as optim
import pandas as pd
import os
import argparse
from data_loader import get_dataloaders
from model import MultiScaleTransformer
from utils import MetricTracker, HuberLoss, visualize_results, visualize_ablation

def train_one_epoch(model, loader, optimizer, criterion_time, criterion_val, lambda_weights, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        seq = batch['seq'].to(device) # (B, L, 2)
        # Target: [Interval, Count, Attacks]
        target = batch['target'].to(device)
        target_time = target[:, 0].unsqueeze(1)
        target_val = target[:, 1:]
        
        optimizer.zero_grad()
        
        pred_time, pred_val, _ = model(seq)
        
        loss_time = criterion_time(pred_time, target_time)
        loss_val = criterion_val(pred_val, target_val)
        
        # Calculate L2 Regularization manually as requested
        # Fix: Use torch.norm for cleaner L2 calculation and correct syntax
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)**2
        
        # Combined Loss
        # Fix: Added l2_reg to the total loss equation
        loss = lambda_weights[0] * loss_time + lambda_weights[1] * loss_val + lambda_weights[2] * l2_reg
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    tracker = MetricTracker()
    total_loss = 0
    criterion_time = torch.nn.MSELoss()
    criterion_val = HuberLoss()
    lambda_weights = [1.0, 1.0, 0.01] # Use same weights for loss reporting

    with torch.no_grad():
        for batch in loader:
            seq = batch['seq'].to(device)
            target = batch['target'].to(device)
            target_time = target[:, 0].unsqueeze(1)
            target_val = target[:, 1:]
            
            pred_time, pred_val, _ = model(seq)
            
            loss_time = criterion_time(pred_time, target_time)
            loss_val = criterion_val(pred_val, target_val)
            
            # Consistent reporting including reg
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)**2
            
            loss = lambda_weights[0] * loss_time + lambda_weights[1] * loss_val + lambda_weights[2] * l2_reg
            total_loss += loss.item()
            
            tracker.update(pred_time, pred_val, target_time, target_val)
            
    metrics = tracker.calculate_metrics()
    metrics['Loss'] = total_loss / len(loader)
    return metrics

def run_experiment(config_name, scales, window_size, data_dir, output_dir, device):
    print(f"\n=== Running Experiment: {config_name} ===")
    print(f"Scales: {scales}, Window: {window_size}")
    
    # 1. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size=32, window_size=window_size, scale_configs=scales
    )
    
    # 2. Model
    model = MultiScaleTransformer(
        scales=scales, 
        window_size=window_size, 
        d_model=64, 
        n_heads=4,
        patch_size=4
    ).to(device)
    
    # 3. Setup
    # Note: Set weight_decay=0 to avoid double regularization since we calculate L2 manually
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    criterion_time = torch.nn.MSELoss()
    criterion_val = HuberLoss(delta=1.0)
    lambda_weights = [1.0, 1.0, 0.01] # Time, Val, Reg
    
    # 4. Loop
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(50): # 50 Epochs
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_time, criterion_val, lambda_weights, device)
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics['Loss']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Early Stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(output_dir, f"{config_name}_best.pth"))
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break
                
    # 5. Test
    model.load_state_dict(torch.load(os.path.join(output_dir, f"{config_name}_best.pth")))
    test_metrics = evaluate(model, test_loader, device)
    
    test_metrics['Config'] = config_name
    test_metrics['Scales'] = str(scales)
    test_metrics['Window'] = window_size
    
    return test_metrics, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/zzzjy/data/IPS')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    all_results = []
    
    # === Experiment A: Scale Combinations ===
    scale_configs = {
        'Scale_1': [1],
        'Scale_1_2': [1, 2],
        'Scale_1_2_4_Base': [1, 2, 4],
        'Scale_1_2_4_8': [1, 2, 4, 8],
        'Scale_1_3': [1, 3],
        'Scale_1_3_9': [1, 3, 9]
    }
    
    best_scale_mae = float('inf')
    best_scale_config = [1, 2, 4] # Default fallback

    for name, scales in scale_configs.items():
        res, _ = run_experiment(name, scales, 48, args.data_dir, args.output_dir, device)
        all_results.append(res)
        if res['MAE'] < best_scale_mae:
            best_scale_mae = res['MAE']
            best_scale_config = scales
    
    print(f"Best Scale Config from Exp A: {best_scale_config} (MAE: {best_scale_mae:.4f})")

    # === Experiment B: Window Sizes ===
    windows = [15, 30, 45, 60, 75, 90]
    base_scales = [1, 2, 4] 
    
    best_window_mae = float('inf')
    best_window = 48

    for w in windows:
        if w == 48: continue # Already run in base
        name = f"Window_{w}"
        res, _ = run_experiment(name, base_scales, w, args.data_dir, args.output_dir, device)
        all_results.append(res)
        if res['MAE'] < best_window_mae:
            best_window_mae = res['MAE']
            best_window = w
            
    # Check if Window 48 (from Exp A Base) was better
    base_res = next((r for r in all_results if r['Config'] == 'Scale_1_2_4_Base'), None)
    if base_res and base_res['MAE'] < best_window_mae:
        best_window = 48
        best_window_mae = base_res['MAE']
        
    print(f"Best Window from Exp B: {best_window} (MAE: {best_window_mae:.4f})")

    # === Experiment C: Combination (Best Scale + Best Window) ===
    # Only run if not already covered by previous experiments
    if best_scale_config != base_scales or best_window != 48:
        print(f"Running Experiment C: Best Scale {best_scale_config} + Best Window {best_window}")
        name = "Exp_C_Best_Comb"
        res, _ = run_experiment(name, best_scale_config, best_window, args.data_dir, args.output_dir, device)
        all_results.append(res)
    else:
        print("Experiment C skipped: Best combination (Base+48) already run in Exp A.")

    # Save CSV
    df = pd.DataFrame(all_results)
    cols = ['Config', 'MAE', 'RMSE', 'SMAPE', 'Peak-MAE', 'Time_Hit_Rate', 'Miss_Rate', 'Total_Hit', 'Total_Miss']
    df = df[cols]
    df.to_csv(os.path.join(args.output_dir, 'ablation_results.csv'), index=False)
    print("\nResults saved to ablation_results.csv")
    print(df)
    
    visualize_ablation(df, args.output_dir)

if __name__ == "__main__":
    main()